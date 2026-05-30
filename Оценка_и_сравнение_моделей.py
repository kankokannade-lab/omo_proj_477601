import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print("1. Загрузка данных и сохранённых моделей...")
df = pd.read_csv("Данные/monsters_features.csv")

# Подготовка X и y (как на предыдущих этапах)
X = df.select_dtypes(include=['float64', 'int64']).drop(columns=['target_class', 'cr_numeric', 'cr'], errors='ignore')
y = df['target_class']
X = X.fillna(0)

print("Воспроизведение тестовой выборки...")
# Фиксируем random_state=42. Нужно получить ту же тестовую выборку
# на которой проверялись модели во время обучения 
# иначе сравнение моделей будет некорректным ¯\_(ツ)_/¯
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Загружаем файлы
dt_model = joblib.load('Модели/dt_best_model.pkl')
knn_model = joblib.load('Модели/knn_best_model.pkl')
knn_scaler = joblib.load('Модели/knn_scaler.pkl')

print("2. Подготовка данных для k-NN...")
# Дерево ест сырые данные, а тестовую выборку для k-NN нужно масштабировать
X_test_scaled = knn_scaler.transform(X_test)

print("3. Получение предсказаний...")
dt_preds = dt_model.predict(X_test)
knn_preds = knn_model.predict(X_test_scaled)

# Текстовые отчеты с основными метриками качества
classes = ['Легкий', 'Средний', 'Опасный', 'Смертоносный']

print("\n" + "="*35)
print("   ОТЧЕТ ДЕРЕВА РЕШЕНИЙ (Decision Tree)")
print("="*35)
print(classification_report(y_test, dt_preds, target_names=classes))

print("\n" + "="*35)
print("   ОТЧЕТ k-NN (k-Nearest Neighbors)")
print("="*35)
print(classification_report(y_test, knn_preds, target_names=classes))


print("\n4. Отрисовка матриц ошибок (Confusion Matrix)...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # Рядом

# Матрица для дерева
cm_dt = confusion_matrix(y_test, dt_preds)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=classes, yticklabels=classes)
axes[0].set_title('Матрица ошибок: Дерево Решений', fontsize=14)
axes[0].set_xlabel('Предсказала модель')
axes[0].set_ylabel('Истинный класс')

# Матрица для k-NN
cm_knn = confusion_matrix(y_test, knn_preds)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Oranges', ax=axes[1], 
            xticklabels=classes, yticklabels=classes)
axes[1].set_title('Матрица ошибок: k-NN', fontsize=14)
axes[1].set_xlabel('Предсказала модель')
axes[1].set_ylabel('Истинный класс')

plt.tight_layout()
plt.savefig("Изображения/confusion_matrices.png", dpi=300)
print("Сравнение завершено. Графики сохранены в 'confusion_matrices.png'.")