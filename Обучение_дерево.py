import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib

print("Шаг 1: Загрузка и очистка данных для Дерева Решений...")
df = pd.read_csv("Данные/monsters_features.csv")

# отделяем признаки X от ответов y
X = df.select_dtypes(include=['float64', 'int64']).drop(columns=['target_class', 'cr_numeric', 'cr']) # сюда 'cr' потому что он ломал всю модель и аккуратность писал рвной 1 (шо бред)
y = df['target_class']

# убираем NaN (как мы делали для k-NN)
X = X.fillna(0)


print("Шаг 2: Разбиение на выборки...")

# используем stratify=y, чтобы сохранить пропорции классов
# random_state=42 оставляем для воспроизводимости результатов
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Шаг 3: Запуск GridSearchCV для Decision Tree... (Масштабирование не требуется!)")

# создаем модель дерева решений
dt = DecisionTreeClassifier(random_state=42)

# сетка параметров:
# max_depth -> ограничение глубины дерева
# min_samples_leaf -> минимальное число объектов в листе
# criterion -> критерий качества разбиения
param_grid_dt = {
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_leaf': [1, 2, 5, 10, 20],
    'criterion': ['gini', 'entropy']
}

# GridSearchCV автоматически перебирает все комбинации параметров
grid_dt = GridSearchCV(
    dt,
    param_grid_dt,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# обучение модели
grid_dt.fit(X_train, y_train)

print("\n   РЕЗУЛЬТАТЫ ДЕРЕВА РЕШЕНИЙ   ")

# вывод лучших параметров
print(f"Лучшие параметры: {grid_dt.best_params_}")

# средняя accuracy по cross-validation
print(f"Точность на обучающей выборке: {grid_dt.best_score_:.4f}")

# дополнительная проверка на тестовой выборке
test_accuracy = grid_dt.score(X_test, y_test)
print(f"Точность на тестовой выборке: {test_accuracy:.4f}")

print("\nШаг 4: Отрисовка схемы дерева...")

# рисуем дерево (๑╹ω╹๑ )
# max_depth=3 ограничивает глубину отображения,
# чтобы схема оставалась читаемой
plt.figure(figsize=(22, 10))

plot_tree(
    grid_dt.best_estimator_,
    feature_names=X.columns,
    class_names=['Легкий', 'Средний', 'Опасный', 'Смертоносный'],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=12
)

plt.title("Визуализация Дерева Решений", fontsize=16)

plt.tight_layout()

# сохраняем изображение дерева
plt.savefig("Изображения/decision_tree_plot.png", dpi=300)

print("Схема дерева сохранена в файл 'decision_tree_plot.png'.")

# сохраняем обученную модель
joblib.dump(grid_dt.best_estimator_, 'Модели/dt_best_model.pkl')

print("Модель Дерева Решений успешно сохранена!")