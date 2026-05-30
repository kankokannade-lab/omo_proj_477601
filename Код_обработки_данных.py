import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Грузим чистые данные
df = pd.read_csv("Данные/monsters_cleaned.csv")

# Для удобного извлечения признаков соединяем все тексты способностей и действий в одну колонку
# Так проще будет проходиться регулярками
text_columns = ['actions', 'special_abilities', 'legendary_actions', 'reactions']
df['full_text'] = ''
for col in text_columns:
    if col in df.columns:
        df['full_text'] += df[col].fillna('').astype(str) + ' '

# ИЗВЛЕЧЕНИЕ МАКС УРОНА (парсинг кубов)

def calculate_max_expected_damage(text):
    """
    Вычисляет максимальное математическое ожидание урона из текстового описания.
    Ищет стандартные паттерны бросков D&D (например, 2d6+4) и считает мат. ожидание.
    """
    if not isinstance(text, str) or text.strip() == '':
        return 0.0
    
    # Ищем формулы вида: XdY, XdY+Z или XdY-Z
    pattern = r'(\d+)d(\d+)(?:\s*([+-])\s*(\d+))?'
    matches = re.findall(pattern, text)
    
    max_damage = 0.0
    for match in matches:
        num_dice = int(match[0])
        faces = int(match[1])
        
        # Мат. ожидание для XdY
        expected_dice_val = num_dice * ((faces + 1) / 2.0)
        
        # Учитываем модиф. урона, если он присутствует
        modifier = 0
        if match[2] and match[3]:
            mod_val = int(match[3])
            modifier = mod_val if match[2] == '+' else -mod_val

        # Нас интересует самая сильная атака, поэтому берём максимум    
        total_expected = expected_dice_val + modifier
        if total_expected > max_damage:
            max_damage = total_expected
            
    return max_damage

df['expected_damage'] = df['full_text'].apply(calculate_max_expected_damage)
print("Признак 'expected_damage' рассчитан.")


# БИНАРНЫЕ МАРКЕРЫ (One-Hot Encoding механик)

# Бинарные флаги для механик, которые сильнее влияют на баланс и CR
mechanics = {
    'has_legendary': 'legendary action',
    'has_multiattack': 'multiattack',
    'has_pack_tactics': 'pack tactics',
    'has_spellcasting': 'spellcasting',
    'has_magic_resistance': 'magic resistance',
    'has_regeneration': 'regeneration'
}

for col_name, keyword in mechanics.items():
    df[col_name] = df['full_text'].apply(lambda x: 1 if keyword in x.lower() else 0)

print("Бинарные флаги механик добавлены.")

# ОЦЕНКА УРОВНЯ МАГИИ

def extract_max_spell_level(text):
    """
    Извлекает макс. уровень заклинания из текста. 
    Заговоры кодируем как 0.5, чтобы модель могла отличить их 
    от полного отсутствия магии (0).
    """
    if not isinstance(text, str):
        return 0
    
    text = text.lower()
    max_level = 0
    
    if 'cantrip' in text:
        max_level = 0.5 
        
    # Ищем упоминания от "1st level" до "9th level"
    pattern = r'(\d)(?:st|nd|rd|th)\s+level'
    levels = re.findall(pattern, text)
    
    if levels:
        max_level_found = max([int(lvl) for lvl in levels])
        if max_level_found > max_level:
            max_level = max_level_found
            
    return max_level

df['max_spell_level'] = df['full_text'].apply(extract_max_spell_level)
print("Признак 'max_spell_level' рассчитан.")


# ВЕКТОРИЗАЦИЯ ТЕКСТА (TF-IDF)

print("Запуск TF-IDF векторизации...")
# Ограничимся 20 самыми важными словами, чтобы не раздувать датасет
tfidf = TfidfVectorizer(max_features=20, stop_words='english')

# Обучаем TF-IDF на нашем тексте
tfidf_matrix = tfidf.fit_transform(df['full_text'])

# Добавляем 'tfidf_' к найденным словам для их отличия
feature_names = [f"tfidf_{word}" for word in tfidf.get_feature_names_out()]

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
df = pd.concat([df, tfidf_df], axis=1)

print("TF-IDF признаки успешно присоединены.")

# ПОДГОТОВКА К ВЫГРУЗКЕ

# Убираем сырой текст, он больше не нужен для ML
df = df.drop(columns=['full_text'])

df.to_csv("monsters_features.csv", index=False)
print("\nДатасет сохранен в 'monsters_features.csv'.")