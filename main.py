"""Desktop application for analysing experimental data from freeze-drying processes."""

import datetime
import os
import sqlite3
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.metrics import r2_score

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QTextDocument
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False


class DataNormalizer:
    """
    Класс для приведения функционально-технологических показателей к единой шкале (0..1).
    Реализует концепцию "привлекательности" из отчета:
    - Выживаемость (Y1) максимизируется (1 - максимум, 0 - минимум).
    - Длительность (Y2) минимизируется (1 - минимум часов, 0 - максимум часов).
    """
    def __init__(self, y1: np.ndarray, y2: np.ndarray):
        self.y1_max = np.max(y1)
        self.y1_min = np.min(y1)
        self.y2_max = np.max(y2)
        self.y2_min = np.min(y2)

    def normalize_y1(self, y: np.ndarray) -> np.ndarray:
        if self.y1_max == self.y1_min:
            return np.zeros_like(y)
        return (y - self.y1_min) / (self.y1_max - self.y1_min)

    def normalize_y2(self, y: np.ndarray) -> np.ndarray:
        if self.y2_max == self.y2_min:
            return np.zeros_like(y)
        return (self.y2_max - y) / (self.y2_max - self.y2_min)


class MathRegressorCore:
    """
    Математическое ядро системы.
    Отвечает за аппроксимацию экспериментальных массивов данных (МНК),
    расчет статистических критериев (R², Фишер), нормализацию функций
    и нахождение оптимального температурного режима сублимации.
    """

    @staticmethod
    def calculate_fisher(r2: float, n: int, k: int) -> float:
        """Расчет экспериментального значения критерия Фишера (F-критерий)"""
        if r2 >= 1.0 or (1.0 - r2) < 1e-7:
            return 999.99 
        if n <= k + 1:
            return 0.0 
        return float((r2 / (1.0 - r2)) * ((n - k - 1) / k))

    @staticmethod
    def find_intersection(x: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> tuple:
        """
        Нахождение точки пересечения двух кривых (нормализованных графиков).
        Возвращает массивы координат пересечений (x_int, y_int).
        """
        diff = y1 - y2
        # Ищем индексы, где разность меняет знак
        idx = np.where(np.diff(np.signbit(diff)))[0]
        
        x_intersects = []
        y_intersects = []
        
        if len(idx) > 0:
            for i in idx:
                x1, x2 = x[i], x[i+1]
                y1_1, y1_2 = y1[i], y1[i+1]
                y2_1, y2_2 = y2[i], y2[i+1]
                
                denominator = ((y1_2 - y1_1) - (y2_2 - y2_1))
                if denominator == 0:
                    continue
                    
                # Точная интерполяция координат пересечения между точками
                x_int = x1 + (x2 - x1) * (y2_1 - y1_1) / denominator
                y_int = y1_1 + (y1_2 - y1_1) * (x_int - x1) / (x2 - x1)
                
                x_intersects.append(x_int)
                y_intersects.append(y_int)
                
        return np.array(x_intersects), np.array(y_intersects)

    def _fit_single_variable(self, x: np.ndarray, y: np.ndarray) -> tuple | None:
        """
        Изолированный подбор наилучшей регрессионной модели для одного массива Y.
        """
        n = len(x)
        results = {}

        # 1. Линейная модель (y = ax + b)
        try:
            p_lin = np.polyfit(x, y, 1)
            y_pred = np.polyval(p_lin, x)
            r2 = r2_score(y, y_pred)
            results['Линейная'] = {
                'coef': p_lin, 'r2': r2, 'fisher': self.calculate_fisher(r2, n, 1),
                'formula': f"y = {p_lin[0]:.4f} * x + {p_lin[1]:.4f}",
                'pred_func': lambda val, p=p_lin: float(p[0] * val + p[1]),
                'plot_func': lambda x_arr, p=p_lin: p[0] * x_arr + p[1]
            }
        except Exception: pass

        # 2. Полиномиальная модель 2-й степени (y = ax² + bx + c)
        try:
            p_poly = np.polyfit(x, y, 2)
            y_pred = np.polyval(p_poly, x)
            r2 = r2_score(y, y_pred)
            results['Полиномиальная'] = {
                'coef': p_poly, 'r2': r2, 'fisher': self.calculate_fisher(r2, n, 2),
                'formula': f"y = {p_poly[0]:.4f}x² + {p_poly[1]:.4f}x + {p_poly[2]:.4f}",
                'pred_func': lambda val, p=p_poly: float(p[0] * (val**2) + p[1] * val + p[2]),
                'plot_func': lambda x_arr, p=p_poly: p[0] * (x_arr**2) + p[1] * x_arr + p[2]
            }
        except Exception: pass

        # 3. Экспоненциальная модель (y = a * e^(bx))
        if np.all(y > 0):
            try:
                p_exp = np.polyfit(x, np.log(y), 1)
                a_val, b_val = np.exp(p_exp[1]), p_exp[0]
                y_pred = a_val * np.exp(b_val * x)
                r2 = r2_score(y, y_pred)
                results['Экспоненциальная'] = {
                    'coef': [a_val, b_val], 'r2': r2, 'fisher': self.calculate_fisher(r2, n, 1),
                    'formula': f"y = {a_val:.4f} * e^({b_val:.4f} * x)",
                    'pred_func': lambda val, a=a_val, b=b_val: float(a * np.exp(b * val)),
                    'plot_func': lambda x_arr, a=a_val, b=b_val: a * np.exp(b * x_arr)
                }
            except Exception: pass

        # 4. Логарифмическая модель (y = a * ln(x) + b)
        if np.all(x > 0):
            try:
                p_log = np.polyfit(np.log(x), y, 1)
                a_val, b_val = p_log[0], p_log[1]
                y_pred = a_val * np.log(x) + b_val
                r2 = r2_score(y, y_pred)
                results['Логарифмическая'] = {
                    'coef': [a_val, b_val], 'r2': r2, 'fisher': self.calculate_fisher(r2, n, 1),
                    'formula': f"y = {a_val:.4f} * ln(x) + {b_val:.4f}",
                    'pred_func': lambda val, a=a_val, b=b_val: float(a * np.log(val) + b if val > 0 else 0.0),
                    'plot_func': lambda x_arr, a=a_val, b=b_val: a * np.log(x_arr) + b_val
                }
            except Exception: pass

        if not results: return None
        optimal_key = max(results, key=lambda k: results[k]['r2'])
        return results, optimal_key

    def analyze_data(self, x_data: list, y1_data: list, y2_data: list) -> dict | None:
        """
        Комплексный расчет:
        - Изолированный поиск лучших уравнений для Y1 и Y2.
        - Вычисление нормализованных кривых на плотной сетке.
        - Поиск пересечения нормализованных кривых (точка оптимума).
        """
        x = np.array(x_data, dtype=float)
        y1 = np.array(y1_data, dtype=float)
        y2 = np.array(y2_data, dtype=float)

        res_y1 = self._fit_single_variable(x, y1)
        res_y2 = self._fit_single_variable(x, y2)

        if not res_y1 or not res_y2:
            return None

        models_y1, best_key_y1 = res_y1
        models_y2, best_key_y2 = res_y2

        best_model_y1 = models_y1[best_key_y1]
        best_model_y2 = models_y2[best_key_y2]

        # Плотная сетка (1000 точек) для построения гладких графиков и точного поиска пересечения
        x_dense = np.linspace(np.min(x), np.max(x), 1000)
        y1_dense = best_model_y1['plot_func'](x_dense)
        y2_dense = best_model_y2['plot_func'](x_dense)

        # Нормализация
        normalizer = DataNormalizer(y1, y2)
        y1_norm = normalizer.normalize_y1(y1_dense)
        y2_norm = normalizer.normalize_y2(y2_dense)

        # Поиск точки пересечения (оптимальной температуры)
        x_intersects, y_intersects = self.find_intersection(x_dense, y1_norm, y2_norm)
        
        optimal_temp = x_intersects[0] if len(x_intersects) > 0 else None
        optimal_norm_val = y_intersects[0] if len(y_intersects) > 0 else None

        return {
            'models_y1': models_y1,
            'best_key_y1': best_key_y1,
            'best_model_y1': best_model_y1,
            
            'models_y2': models_y2,
            'best_key_y2': best_key_y2,
            'best_model_y2': best_model_y2,
            
            'x_dense': x_dense,
            'y1_dense': y1_dense,
            'y2_dense': y2_dense,
            'y1_norm': y1_norm,
            'y2_norm': y2_norm,
            
            'optimal_temperature': optimal_temp,
            'optimal_norm_val': optimal_norm_val,
            'normalizer': normalizer
        }


class LocalDatabaseManager:
    """
    Класс для управления локальной базой данных SQLite.
    Отвечает за инициализацию БД, сохранение истории комплексных экспериментов
    и загрузку трехмерных массивов точек (X, Y1, Y2).
    """
    def __init__(self, db_name="local_experiments.db"):
        # Жестко вычисляем абсолютный путь к папке, где лежит main.py
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(base_dir, db_name)
        
        self.init_db()

    def init_db(self):
        """Инициализация таблиц локальной БД с поддержкой трех технологических переменных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Таблица метаданных и агрегированных результатов экспериментов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                r2_y1 REAL,
                r2_y2 REAL,
                optimal_temp REAL
            )
        """)
        
        # Обновленная таблица точек: теперь хранит x_val, y1_val и y2_val
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiment_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                x_val REAL NOT NULL,
                y1_val REAL NOT NULL,
                y2_val REAL NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
            )
        """)
        conn.commit()
        conn.close()

    def save_experiment(self, product_name, x_vals, y1_vals, y2_vals, r2_y1=0.0, r2_y2=0.0, optimal_temp=None):
        """
        Сохранение комплексного эксперимента и всех его измерительных точек.
        Принимает массивы x_vals, y1_vals, y2_vals.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Запись метаданных
        cursor.execute("""
            INSERT INTO experiments (product_name, timestamp, r2_y1, r2_y2, optimal_temp)
            VALUES (?, ?, ?, ?, ?)
        """, (product_name, timestamp, r2_y1, r2_y2, optimal_temp))
        
        experiment_id = cursor.lastrowid
        
        # Построчная запись измерительных векторов в обновленную структуру таблицы
        for x, y1, y2 in zip(x_vals, y1_vals, y2_vals):
            cursor.execute("""
                INSERT INTO experiment_points (experiment_id, x_val, y1_val, y2_val)
                VALUES (?, ?, ?, ?)
            """, (experiment_id, float(x), float(y1), float(y2)))
            
        conn.commit()
        conn.close()
        return experiment_id

    def load_all_experiments(self):
        """Получение списка всех экспериментов для диалогового окна выбора"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, product_name, timestamp, r2_y1 FROM experiments ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()
        return rows

    def load_experiment_points(self, experiment_id):
        """
        Загрузка измерительных точек по ID эксперимента.
        Возвращает кортеж из трех массивов (x_vals, y1_vals, y2_vals).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT x_val, y1_val, y2_val 
            FROM experiment_points 
            WHERE experiment_id = ? 
            ORDER BY id ASC
        """, (experiment_id,))
        rows = cursor.fetchall()
        conn.close()
        
        x_vals = [row[0] for row in rows]
        y1_vals = [row[1] for row in rows]
        y2_vals = [row[2] for row in rows]
        
        return x_vals, y1_vals, y2_vals


class GlobalDatabaseManager:
    """Интеграция с корпоративной сервером СУБД MySQL через Workbench"""
    def __init__(self):
        self.config = {
            'host': os.getenv('MYSQL_HOST', 'localhost'),
            'user': os.getenv('MYSQL_USER', 'root'),
            'password': os.getenv('MYSQL_PASSWORD', ''),
            'database': os.getenv('MYSQL_DATABASE', 'central_formulas')
        }

    def fetch_global_products(self) -> list:
        """Получение синхронизированного списка продуктов для ComboBox"""
        if not MYSQL_AVAILABLE:
            return ["Драйвер MySQL отсутствует"]
        try:
            conn = mysql.connector.connect(**self.config)
            cursor = conn.cursor()
            cursor.execute("SELECT product_name FROM products ORDER BY product_name")
            products = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            return products
        except Exception:
            return []

    def upload_reference_model(self, product_name: str, dep_type: str, 
                               formula_y1: str, r2_y1: float,
                               formula_y2: str, r2_y2: float,
                               optimal_temp: float | None) -> tuple:
        """Отправка полного пакета данных (2 модели + оптимум) на сервер MySQL"""
        if not MYSQL_AVAILABLE: 
            return False, "Драйвер mysql-connector не установлен."
            
        try:
            conn = mysql.connector.connect(**self.config)
            c = conn.cursor()
            
            # Проверяем, есть ли продукт в справочнике
            c.execute("SELECT product_id FROM products WHERE product_name = %s", (product_name,))
            row = c.fetchone()
            if not row:
                c.execute("INSERT INTO products (product_name) VALUES (%s)", (product_name,))
                prod_id = c.lastrowid
            else:
                prod_id = row[0]
                
            # Сохраняем комплексный эталон
            c.execute("""
                INSERT INTO reference_models 
                (product_id, dependency_type, formula_y1, r2_y1, formula_y2, r2_y2, optimal_temp) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (prod_id, dep_type, formula_y1, r2_y1, formula_y2, r2_y2, optimal_temp))
            
            conn.commit()
            conn.close()
            return True, "Эталон успешно синхронизирован с сервером MySQL."
            
        except Exception as e: 
            return False, f"Ошибка MySQL: {str(e)}"

class ExperimentSelectionDialog(QDialog):
    """Диалоговое окно для выбора эксперимента из базы данных"""
    def __init__(self, experiments, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выберите эксперимент для загрузки")
        self.resize(500, 300)
        self.selected_id = None

        layout = QVBoxLayout(self)

        self.list_widget = QListWidget()
        for exp in experiments:
            # exp имеет структуру: (id, product_name, timestamp, r2)
            exp_id, prod_name, timestamp, r2 = exp
            item_text = f"ID: {exp_id} | {prod_name} | {timestamp} | R²: {r2:.4f}"
            self.list_widget.addItem(item_text)
            
            # Сохраняем ID эксперимента в скрытые данные элемента списка
            self.list_widget.item(self.list_widget.count() - 1).setData(Qt.UserRole, exp_id)

        layout.addWidget(self.list_widget)

        # Кнопки Ок и Отмена
        self.btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_box.accepted.connect(self.accept)
        self.btn_box.rejected.connect(self.reject)
        layout.addWidget(self.btn_box)

    def accept(self):
        selected_items = self.list_widget.selectedItems()
        if selected_items:
            self.selected_id = selected_items[0].data(Qt.UserRole)
        super().accept()


class ApplicationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.core = MathRegressorCore()
        self.local_db = LocalDatabaseManager()
        self.global_db = GlobalDatabaseManager()
        
        self.current_results = None
        self.loaded_x = []
        self.loaded_y1 = []
        self.loaded_y2 = []
        self.forecast_point = None
        
        self.init_ui()
        self.load_products()

    def init_ui(self):
        self.setWindowTitle("Информационная система анализа сублимационной сушки")
        self.resize(1400, 850)
        self.setFont(QFont("Segoe UI", 10))
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- ЛЕВАЯ ПАНЕЛЬ (Управление) ---
        control_panel = QVBoxLayout()
        main_layout.addLayout(control_panel, stretch=2)
        
        # 1. Продукция
        prod_group = QGroupBox("1. Выбор исследуемого продукта")
        prod_layout = QVBoxLayout(prod_group)
        self.combo_products = QComboBox()
        self.load_products()
        prod_layout.addWidget(self.combo_products)
        control_panel.addWidget(prod_group)
        
        # 2. Данные и Импорт
        data_group = QGroupBox("2. Экспериментальные данные")
        data_layout = QVBoxLayout(data_group)
        
        btn_excel = QPushButton("Импорт из Excel (.xlsx)")
        btn_excel.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        btn_excel.clicked.connect(self.import_from_excel)
        data_layout.addWidget(btn_excel)
        
        self.table_data = QTableWidget()
        self.table_data.setColumnCount(3)
        self.table_data.setHorizontalHeaderLabels([
            "X (Температура, °C)", 
            "Y1 (Выживаемость, %)", 
            "Y2 (Длительность, ч)"
        ])
        self.table_data.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_data.setRowCount(5)
        data_layout.addWidget(self.table_data)
        
        table_btns = QHBoxLayout()
        btn_add = QPushButton("+ Строка")
        btn_add.clicked.connect(lambda: self.table_data.setRowCount(self.table_data.rowCount() + 1))
        btn_rem = QPushButton("- Строка")
        btn_rem.clicked.connect(lambda: self.table_data.setRowCount(max(1, self.table_data.rowCount() - 1)))
        btn_clear = QPushButton("Очистить")
        btn_clear.clicked.connect(self.clear_table)
        for b in (btn_add, btn_rem, btn_clear): table_btns.addWidget(b)
        data_layout.addLayout(table_btns)
        
        self.btn_calc = QPushButton("ВЫПОЛНИТЬ РАСЧЕТ")
        self.btn_calc.setStyleSheet("background-color: #1976D2; color: white; font-weight: bold; padding: 10px;")
        self.btn_calc.clicked.connect(self.process_calculation)
        data_layout.addWidget(self.btn_calc)
        control_panel.addWidget(data_group)
        
        # 3. Результаты
        res_group = QGroupBox("3. Результаты анализа")
        res_layout = QVBoxLayout(res_group)
        self.lbl_opt_temp = QLabel("Оптимальная температура: —")
        self.lbl_opt_temp.setStyleSheet("color: #D32F2F; font-weight: bold; font-size: 14px;")
        self.lbl_model_y1 = QLabel("Y1 (Выживаемость): —")
        self.lbl_model_y2 = QLabel("Y2 (Длительность): —")
        for lbl in (self.lbl_opt_temp, self.lbl_model_y1, self.lbl_model_y2):
            res_layout.addWidget(lbl)
        export_btns_layout = QHBoxLayout()
        
        self.btn_export_chart = QPushButton("Экспорт графика (PNG)")
        self.btn_export_chart.setStyleSheet("background-color: #009688; color: white; font-weight: bold; padding: 8px;")
        self.btn_export_chart.clicked.connect(self.export_current_chart)
        
        self.btn_export_report = QPushButton("Экспорт отчета (PDF/TXT)")
        self.btn_export_report.setStyleSheet("background-color: #607D8B; color: white; font-weight: bold; padding: 8px;")
        self.btn_export_report.clicked.connect(self.export_analytical_report)
        
        export_btns_layout.addWidget(self.btn_export_chart)
        export_btns_layout.addWidget(self.btn_export_report)
        res_layout.addLayout(export_btns_layout)
        control_panel.addWidget(res_group)
        
        
        # 4. Базы Данных
        db_group = QGroupBox("4. Работа с Базами Данных")
        db_layout = QVBoxLayout(db_group)
        db_btns_top = QHBoxLayout()
        btn_save_local = QPushButton("Сохранить в SQLite")
        btn_save_local.clicked.connect(self.save_to_local)
        btn_load_local = QPushButton("Загрузить из SQLite")
        btn_load_local.clicked.connect(self.load_from_local)
        db_btns_top.addWidget(btn_save_local)
        db_btns_top.addWidget(btn_load_local)
        btn_sync_global = QPushButton("Синхронизировать эталон (MySQL)")
        btn_sync_global.setStyleSheet("background-color: #FF9800; font-weight: bold;")
        btn_sync_global.clicked.connect(self.sync_to_global)
        db_layout.addLayout(db_btns_top)
        db_layout.addWidget(btn_sync_global)
        control_panel.addWidget(db_group)

        # --- ПРАВАЯ ПАНЕЛЬ (Графики Matplotlib) ---
        plot_group = QGroupBox("Визуализация математических моделей")
        plot_layout = QVBoxLayout(plot_group)
        self.figure = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        main_layout.addWidget(plot_group, stretch=3)

    # --- ЛОГИКА ---
    def load_products(self):
        """Загрузка номенклатуры из MySQL или использование локального резервного списка"""
        # Очищаем выпадающий список от старых или дефолтных значений
        self.combo_products.clear()
        
        products = None
        try:
            if MYSQL_AVAILABLE and self.global_db:
                products = self.global_db.fetch_global_products()
        except Exception:
            products = None

        # Полная номенклатура из вашего дипломного проекта
        fallback_products = [
            "Йогурт без добавок",
            "Йогурт с 10% сладкого пюре",
            "Йогурт с 15% сладкого пюре",
            "Йогурт с 20% сладкого пюре",
            "Йогурт с 10% кислого пюре",
            "Йогурт с 15% кислого пюре",
            "Йогурт с 20% кислого пюре",
            "Йогурт с 10% овощного пюре",
            "Йогурт с 15% овощного пюре",
            "Йогурт с 20% овощного пюре",
            "Биойогурт без добавок",
            "Биойогурт с 10% сладкого пюре",
            "Биойогурт с 15% сладкого пюре",
            "Биойогурт с 20% сладкого пюре",
            "Биойогурт с 10% кислого пюре",
            "Биойогурт с 15% кислого пюре",
            "Биойогурт с 20% кислого пюре",
            "Биойогурт с 10% овощного пюре",
            "Биойогурт с 15% овощного пюре",
            "Биойогурт с 20% овощного пюре",
            "Простокваша без добавок"
        ]

        if (
            products
            and isinstance(products, list)
            and products[0] != "Драйвер MySQL отсутствует"
        ):
            self.combo_products.addItems(products)
        else:
            self.combo_products.addItems(fallback_products)

    def clear_table(self):
        self.table_data.setRowCount(0)
        self.table_data.setRowCount(5)

    def import_from_excel(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Открыть файл данных", "", "Excel Files (*.xlsx *.xls);;CSV Files (*.csv)")
        if not file_path: return
        try:
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
            if df.shape[1] < 3:
                QMessageBox.warning(self, "Ошибка", "Файл должен содержать минимум 3 колонки: X, Y1, Y2.")
                return
            self.table_data.setRowCount(len(df))
            for row_idx, row in df.iterrows():
                for col_idx in range(3):
                    val = row.iloc[col_idx]
                    val_str = str(val).replace(',', '.') if not pd.isna(val) else ""
                    self.table_data.setItem(row_idx, col_idx, QTableWidgetItem(val_str))
            QMessageBox.information(self, "Успех", "Данные импортированы!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def process_calculation(self):
        x_vals, y1_vals, y2_vals = [], [], []
        for i in range(self.table_data.rowCount()):
            item_x = self.table_data.item(i, 0)
            item_y1 = self.table_data.item(i, 1)
            item_y2 = self.table_data.item(i, 2)
            
            txt_x = item_x.text().strip() if item_x else ""
            txt_y1 = item_y1.text().strip() if item_y1 else ""
            txt_y2 = item_y2.text().strip() if item_y2 else ""
            
            if not txt_x and not txt_y1 and not txt_y2: continue
            if not txt_x or not txt_y1 or not txt_y2:
                QMessageBox.warning(self, "Ошибка", f"Строка {i + 1} заполнена не полностью!")
                return
                
            try:
                x_vals.append(float(txt_x.replace(',', '.')))
                y1_vals.append(float(txt_y1.replace(',', '.')))
                y2_vals.append(float(txt_y2.replace(',', '.')))
            except ValueError:
                QMessageBox.warning(self, "Ошибка", f"Некорректный формат чисел в строке {i + 1}.")
                return

        if len(x_vals) < 3:
            QMessageBox.warning(self, "Ошибка", "Необходимо минимум 3 точки.")
            return

        self.loaded_x = x_vals
        self.loaded_y1 = y1_vals
        self.loaded_y2 = y2_vals
        
        # Вызов нового математического ядра
        res = self.core.analyze_data(x_vals, y1_vals, y2_vals)
        if res is None:
            QMessageBox.critical(self, "Ошибка", "Не удалось аппроксимировать данные.")
            return
            
        self.current_results = res
        
        # Обновление UI
        m1 = res['best_model_y1']
        m2 = res['best_model_y2']
        self.lbl_model_y1.setText(f"Y1 ({res['best_key_y1']}): {m1['formula']} (R²: {m1['r2']:.3f})")
        self.lbl_model_y2.setText(f"Y2 ({res['best_key_y2']}): {m2['formula']} (R²: {m2['r2']:.3f})")
        
        if res['optimal_temperature'] is not None:
            self.lbl_opt_temp.setText(f"Оптимальная температура: {res['optimal_temperature']:.2f} °C")
        else:
            self.lbl_opt_temp.setText("Оптимальная температура: Не найдена (нет пересечения)")
            
        self.draw_plot()

    def draw_plot(self):
        self.figure.clear()
        res = self.current_results
        x_arr = np.array(self.loaded_x)
        y1_arr = np.array(self.loaded_y1)
        y2_arr = np.array(self.loaded_y2)
        
        # График 1: Выживаемость
        ax1 = self.figure.add_subplot(311)
        ax1.scatter(x_arr, y1_arr, color='black', label='Данные (Выживаемость)')
        ax1.plot(res['x_dense'], res['y1_dense'], color='#1976D2', label=f"{res['best_key_y1']} модель")
        ax1.set_ylabel("Выживаемость, %")
        ax1.grid(True, linestyle=':')
        ax1.legend()
        
        # График 2: Длительность
        ax2 = self.figure.add_subplot(312)
        ax2.scatter(x_arr, y2_arr, color='black', label='Данные (Длительность)')
        ax2.plot(res['x_dense'], res['y2_dense'], color='#388E3C', label=f"{res['best_key_y2']} модель")
        ax2.set_ylabel("Длительность, ч")
        ax2.grid(True, linestyle=':')
        ax2.legend()
        
        # График 3: Нормализованные функции и оптимум
        ax3 = self.figure.add_subplot(313)
        ax3.plot(res['x_dense'], res['y1_norm'], color='blue', label='Норм. выживаемость')
        ax3.plot(res['x_dense'], res['y2_norm'], color='red', label='Норм. длительность')
        
        if res['optimal_temperature'] is not None:
            opt_x = res['optimal_temperature']
            opt_y = res['optimal_norm_val']
            ax3.plot(opt_x, opt_y, marker='o', color='black', markersize=8, label=f"Оптимум: {opt_x:.2f}°C")
            
            # Аннотация
            real_y1 = np.interp(opt_x, res['x_dense'], res['y1_dense'])
            real_y2 = np.interp(opt_x, res['x_dense'], res['y2_dense'])
            ax3.text(0.02, 0.95, f"T = {opt_x:.2f} °C\nВыживаемость: {real_y1:.1f} %\nДлительность: {real_y2:.1f} ч", 
                     transform=ax3.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
        ax3.set_xlabel("Температура сублимации, °C")
        ax3.set_ylabel("Нормированное значение")
        ax3.grid(True, linestyle=':')
        ax3.legend()
        
        self.figure.tight_layout()
        self.canvas.draw()

    def save_to_local(self):
        """Сохранение результатов текущего расчета в локальную БД"""
        # Проверяем, был ли вообще проведен расчет
        if not hasattr(self, 'current_results') or not self.current_results:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения. Сначала выполните расчет.")
            return

        try:
            # Read the current values directly from the table.
            x_vals, y1_vals, y2_vals = [], [], []
            for row in range(self.table_data.rowCount()):
                item_x = self.table_data.item(row, 0)
                item_y1 = self.table_data.item(row, 1)
                item_y2 = self.table_data.item(row, 2)
                
                # Если ячейка X существует и не пустая
                if item_x and item_x.text().strip():
                    x_vals.append(float(item_x.text().replace(',', '.')))
                    y1_vals.append(float(item_y1.text().replace(',', '.')))
                    y2_vals.append(float(item_y2.text().replace(',', '.')))

            # Extract calculated metadata.
            res = self.current_results
            r2_y1 = 0.0
            r2_y2 = 0.0
            opt_temp = None

            try:
                r2_y1 = res.get('best_model_y1', {}).get('r2', 0.0)
                r2_y2 = res.get('best_model_y2', {}).get('r2', 0.0)
            except Exception:
                pass

            opt_temp = res.get('optimal_temperature')

            # Save the experiment.
            exp_id = self.local_db.save_experiment(
                product_name=self.combo_products.currentText(),
                x_vals=x_vals,
                y1_vals=y1_vals,
                y2_vals=y2_vals,
                r2_y1=r2_y1,
                r2_y2=r2_y2,
                optimal_temp=opt_temp
            )
            QMessageBox.information(
                self, "SQLite", 
                f"Эксперимент успешно сохранен! ID записи: {exp_id}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка SQLite", 
                f"Не удалось сохранить данные: {str(e)}"
            )

    def load_from_local(self):
        """Загрузка выбранного пользователем эксперимента из SQLite и заполнение UI таблицы"""
        experiments = self.local_db.load_all_experiments()
        if not experiments:
            QMessageBox.information(self, "Пусто", "В локальной базе нет сохраненных расчетов.")
            return

        # Вызов диалогового окна выбора исторического расчета
        dialog = ExperimentSelectionDialog(experiments, self)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_id is not None:
            try:
                # Извлекаем полноценный кортеж из трех массивов данных
                x_vals, y1_vals, y2_vals = self.local_db.load_experiment_points(dialog.selected_id)
                
                # Подготовка интерфейсной таблицы
                self.clear_table()
                self.table_data.setRowCount(len(x_vals))
                
                # Заполнение всех 3 колонок без использования заглушек
                for i, (x, y1, y2) in enumerate(zip(x_vals, y1_vals, y2_vals)):
                    self.table_data.setItem(i, 0, QTableWidgetItem(str(x)))
                    self.table_data.setItem(i, 1, QTableWidgetItem(str(y1)))
                    self.table_data.setItem(i, 2, QTableWidgetItem(str(y2)))
                    
                QMessageBox.information(
                    self, "Локальная БД", 
                    f"Эксперимент (ID: {dialog.selected_id}) успешно загружен во все технологические поля."
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Ошибка СУБД", 
                    f"Критическая ошибка при чтении структуры данных: {str(e)}"
                )

    def sync_to_global(self):
        """Сбор текущих рассчитанных данных и отправка их в глобальную БД"""
        if not self.current_results: 
            QMessageBox.warning(self, "Внимание", "Сначала выполните расчет!")
            return
            
        res = self.current_results
        
        # Извлекаем формулы и R² для обеих моделей
        f_y1 = res['best_model_y1']['formula']
        r2_y1 = res['best_model_y1']['r2']
        
        f_y2 = res['best_model_y2']['formula']
        r2_y2 = res['best_model_y2']['r2']
        
        # Извлекаем точку оптимума (может быть None, если графики не пересеклись)
        opt_temp = res['optimal_temperature']
        
        # Передаем полный пакет данных в менеджер БД
        ok, msg = self.global_db.upload_reference_model(
            product_name=self.combo_products.currentText(), 
            dep_type="Комплексный анализ",
            formula_y1=f_y1, 
            r2_y1=r2_y1,
            formula_y2=f_y2,
            r2_y2=r2_y2,
            optimal_temp=opt_temp
        )
        
        if ok: 
            QMessageBox.information(self, "MySQL", msg)
        else: 
            QMessageBox.critical(self, "Ошибка MySQL", msg)

    def export_current_chart(self):
        """Экспорт текущего графического холста в файл изображения PNG (высокое качество для печати)"""
        if not self.current_results:
            QMessageBox.warning(self, "Экспорт", "Нет данных для экспорта. Сначала выполните расчет.")
            return
            
        # Формируем имя файла из названия продукта
        product_name = self.combo_products.currentText().replace(" ", "_").replace("%", "")
        default_name = f"Графики_оптимизации_{product_name}.png"
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить график", default_name, "Images (*.png)")
        if file_path:
            try:
                # 300 dpi - ГОСТовский стандарт типографской печати для ВКР
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Успех", f"Графики успешно сохранены в:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить график: {str(e)}")

    def export_analytical_report(self):
        """Генерация комплексного отчета в формате PDF через встроенный QPrinter"""
        if not self.current_results:
            QMessageBox.warning(self, "Экспорт", "Нет результатов анализа для формирования отчета.")
            return
            
        product = self.combo_products.currentText()
        res = self.current_results
        
        default_name = f"Технологический_отчет_{product.replace(' ', '_')}.pdf"
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить отчет", default_name, "PDF Document (*.pdf);;Text File (*.txt)")
        
        if not file_path:
            return
            
        try:
            # Подготовка данных оптимума
            if res['optimal_temperature'] is not None:
                real_y1 = np.interp(res['optimal_temperature'], res['x_dense'], res['y1_dense'])
                real_y2 = np.interp(res['optimal_temperature'], res['x_dense'], res['y2_dense'])
                optimum_html = f"""
                    <h3 style="color: #D32F2F;">КРИТИЧЕСКАЯ ТОЧКА ОПТИМУМА: {res['optimal_temperature']:.2f} &deg;C</h3>
                    <p>При данной температуре достигаются сбалансированные параметры:</p>
                    <ul>
                        <li>Ожидаемая выживаемость микрофлоры: <b>{real_y1:.2f} %</b></li>
                        <li>Ожидаемая длительность сушки: <b>{real_y2:.2f} ч.</b></li>
                    </ul>
                """
            else:
                optimum_html = "<p style='color: red;'><b>Внимание:</b> Точка комплексного технологического оптимума не найдена.</p>"

            # Создание HTML-каркаса для красивого документа
            html_content = f"""
            <html>
            <head><style>body {{ font-family: Arial, sans-serif; line-height: 1.5; }} th, td {{ padding: 8px; border: 1px solid black; text-align: center; }} table {{ border-collapse: collapse; width: 100%; }}</style></head>
            <body>
                <h1 align="center">НАУЧНО-ТЕХНОЛОГИЧЕСКИЙ ОТЧЕТ</h1>
                <h2 align="center">Моделирование процесса сублимационной сушки</h2>
                <hr>
                <p><b>Исследуемый объект/продукция:</b> {product}</p>
                <p><b>Дата генерации отчета:</b> {datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</p>
                
                <h3>1. Исходные экспериментальные данные</h3>
                <table>
                    <tr><th>Температура (X, &deg;C)</th><th>Выживаемость (Y1, %)</th><th>Длительность (Y2, ч)</th></tr>
            """
            
            # Заполнение таблицы данными
            for x, y1, y2 in zip(self.loaded_x, self.loaded_y1, self.loaded_y2):
                html_content += f"<tr><td>{x}</td><td>{y1}</td><td>{y2}</td></tr>"
                
            html_content += f"""
                </table>
                
                <h3>2. Результаты математического моделирования</h3>
                <p><b>Показатель Y1 (Выживаемость бактерий):</b></p>
                <ul>
                    <li>Оптимальная модель: {res['best_key_y1']}</li>
                    <li>Уравнение регрессии: <i>{res['best_model_y1']['formula']}</i></li>
                    <li>Коэффициент детерминации (R&sup2;): {res['best_model_y1']['r2']:.4f}</li>
                    <li>Критерий Фишера: {res['best_model_y1']['fisher']:.2f}</li>
                </ul>
                
                <p><b>Показатель Y2 (Длительность процесса):</b></p>
                <ul>
                    <li>Оптимальная модель: {res['best_key_y2']}</li>
                    <li>Уравнение регрессии: <i>{res['best_model_y2']['formula']}</i></li>
                    <li>Коэффициент детерминации (R&sup2;): {res['best_model_y2']['r2']:.4f}</li>
                    <li>Критерий Фишера: {res['best_model_y2']['fisher']:.2f}</li>
                </ul>
                <hr>
                
                <h3>3. Технологический вердикт системы</h3>
                {optimum_html}
                <hr>
                <p align="center" style="font-size: 10px; color: grey;">Отчет сформирован автоматически Информационной Системой</p>
            </body>
            </html>
            """

            # В зависимости от расширения файла сохраняем PDF или обычный TXT
            if file_path.endswith('.pdf'):
                document = QTextDocument()
                document.setHtml(html_content)
                printer = QPrinter(QPrinter.HighResolution)
                printer.setOutputFormat(QPrinter.PdfFormat)
                printer.setOutputFileName(file_path)
                # Устанавливаем отступы страницы
                printer.setPageMargins(15, 15, 15, 15, QPrinter.Millimeter)
                document.print_(printer)
            else:
                # Очистка HTML тегов для создания простого текстового файла
                import re
                clean_text = re.sub('<[^<]+>', '', html_content.replace('</tr>', '\n').replace('</td>', '\t').replace('<li>', ' * ').replace('<br>', '\n'))
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(clean_text)

            QMessageBox.information(self, "Успех", f"Отчет успешно сохранен в:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить отчет: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ApplicationWindow()
    window.show()
    sys.exit(app.exec_())






