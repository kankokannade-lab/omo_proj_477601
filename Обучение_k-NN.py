import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

print("Загрузка и очистка данных для k-NN...")
df = pd.read_csv("Данные/monsters_features.csv")

# Формируем матрицу признаков (X) и вектор целевой переменной (y)
# Оставляем только числовые фичи, исключая таргеты и служебные столбцы
X = df.select_dtypes(include=['float64', 'int64']).drop(columns=['target_class', 'cr_numeric', 'cr']) 
y = df['target_class']

# Пропуски нулями (для согласования с деревом)
X = X.fillna(0)

print("Разбиение на выборки...")
# Используем те же параметры, что и в дереве, для одинакового разделения
# Фиксируем random_state для воспроизводимости.
# stratify=y, чтобы сохранить исходный баланс классов 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Масштабирование данных...")
scaler = StandardScaler()

# ВАЖНО (на всякий случай): Обучаем скейлер только на тренировочной выборке, чтобы избежать утечки данных из теста в трейн.
X_train_scaled = scaler.fit_transform(X_train)
# Тестовую выборку только трансформируем
X_test_scaled = scaler.transform(X_test)

print("Запуск кросс-валидации и поиска гиперпараметров...")
knn = KNeighborsClassifier()

# Сетка параметров: перебор соседей от 1 до 20 и метрик расстояния
param_grid_knn = {
    'n_neighbors': range(1, 21),
    'metric': ['euclidean', 'manhattan'],
    'weights': ['uniform', 'distance'] # Добавили веса для чуть более тонкой настройки
}

# GridSearchCV для поиска лучших параметров
grid_knn = GridSearchCV(
    knn,
    param_grid_knn,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Собственно, обучение на масштабированных данных:
grid_knn.fit(X_train_scaled, y_train)

print("\nИтоги обучения k-NN:")
print(f"Лучшие параметры: {grid_knn.best_params_}")
print(f"Точность на обучающей выборке (CV): {grid_knn.best_score_:.4f}")

# Проверка на тестовой выборке
test_accuracy = grid_knn.score(X_test_scaled, y_test)
print(f"Точность на тестовой выборке: {test_accuracy:.4f}")

print("\nСохранение модели и скейлера...")
joblib.dump(grid_knn.best_estimator_, 'Модели/knn_best_model.pkl')
joblib.dump(scaler, 'Модели/knn_scaler.pkl')

print("Модель и Scaler успешно сохранены.")