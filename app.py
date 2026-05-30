import streamlit as st
import pandas as pd
import joblib

# Подгружаем лучшую модель (в нашем случае это дерево, т. к. 
# оно работает с сырыми данными и не нужно тащить сюда StandardScaler)
model = joblib.load('Модели/dt_best_model.pkl')

st.set_page_config(page_title="D&D Threat Analyzer", page_icon="🐉", layout="wide")

st.title("🐉 Умный Бестиарий: Относительная Угроза")
st.write("Этот ИИ не просто оценивает абсолютную силу монстра, но и рассчитывает его опасность **специально для вашей группы**.")


# НАСТРОЙКИ ПАРТИИ (сайдбар слева)
# Даём возможность задать контекст для оценки угрозы
st.sidebar.header("🛡️ Настройки вашей группы")
party_size = st.sidebar.number_input("Количество игроков", min_value=1, max_value=10, value=4)
party_level = st.sidebar.slider("Средний уровень игроков", min_value=1, max_value=20, value=3)

# Официальные пороги опыта (XP) на 1 персонажа по правилам D&D 5e (DMG)
# Структура: Уровень: [Легкий, Средний, Сложный, Смертоносный]
xp_thresholds_per_player = {
    1: [25, 50, 75, 100],       2: [50, 100, 150, 200],
    3: [75, 150, 225, 400],     4: [125, 250, 375, 500],
    5: [250, 500, 750, 1100],   6: [300, 600, 900, 1400],
    7: [350, 750, 1100, 1700],  8: [450, 900, 1400, 2100],
    9: [500, 1100, 1600, 2400], 10: [600, 1200, 1900, 2800],
    11: [800, 1600, 2400, 3600],12: [1000, 2000, 3000, 4500],
    13: [1100, 2200, 3400, 5100],14: [1250, 2500, 3800, 5700],
    15: [1400, 2800, 4300, 6400],16: [1600, 3200, 4800, 7200],
    17: [2000, 3900, 5900, 8800],18: [2100, 4200, 6300, 9500],
    19: [2400, 4900, 7300, 10900],20: [2800, 5700, 8500, 12700]
}

# Рассчитываем пороги для всей группы (умножаем на количество игроков)
party_thresholds = [x * party_size for x in xp_thresholds_per_player[party_level]]

# ВВОД СТАТОВ МОНСТРА (центр)

with st.form("monster_form"):
    st.subheader("Характеристики сканируемого монстра")
    col1, col2 = st.columns(2)
    
    with col1:
        hp = st.number_input("Hit Points (Здоровье)", min_value=1, max_value=1500, value=50)
        ac = st.number_input("Armor Class (Броня)", min_value=1, max_value=30, value=15)
        exp_damage = st.number_input("Ожидаемый урон за ход", min_value=0, max_value=500, value=15)
        
    with col2:
        max_spell_level = st.slider("Максимальный уровень магии", 0, 9, 0)
        has_legendary = st.checkbox("Есть Легендарные Действия")
        
    submit_button = st.form_submit_button("Оценить относительную угрозу")


# ЛОГИКА И ВЫВОД РЕЗУЛЬТАТА

if submit_button:
    # Формируем датафрейм с нулями, опираясь на структуру признаков, 
    # которую модель запомнила при обучении.
    expected_columns = model.feature_names_in_
    input_data = pd.DataFrame(0, index=[0], columns=expected_columns)
    
    # Заполнение данных (только те признаки, которые ввёл пользователь)
    if 'armor_class' in input_data.columns: input_data['armor_class'] = ac
    if 'hit_points' in input_data.columns: input_data['hit_points'] = hp
    elif 'hp' in input_data.columns: input_data['hp'] = hp
    if 'expected_damage' in input_data.columns: input_data['expected_damage'] = exp_damage
    if 'has_legendary' in input_data.columns: input_data['has_legendary'] = 1 if has_legendary else 0
    if 'max_spell_level' in input_data.columns: input_data['max_spell_level'] = max_spell_level

    # Получаем сырой ответ от модели (абсолютный класс угрозы 0, 1, 2 или 3)
    raw_prediction = model.predict(input_data)[0]
    
    # Переводим предсказания в целые числа (вдруг модель вернула '0')
    raw_prediction = int(raw_prediction) if str(raw_prediction).isdigit() else 0

    # Присваиваем монстру примерное количество XP, в зависимости от его абсолютного тира
    # (Цифры основаны на средних значениях из D&D)
    monster_xp_map = {
        0: 200,    # ~ CR 1
        1: 2300,   # ~ CR 6
        2: 11500,  # ~ CR 14
        3: 25000   # ~ CR 20+
    }
    
    monster_xp = monster_xp_map.get(raw_prediction, 200)

    st.markdown("---")
    st.subheader("📊 Анализ Энкаунтера")
    
    st.write(f"**Анализ ИИ:** Модель классифицировала этого монстра как Тир {raw_prediction} (Примерно {monster_xp} XP).")
    st.write(f"**Пороги вашей группы:** Легко ({party_thresholds[0]} XP) | Средне ({party_thresholds[1]} XP) | Сложно ({party_thresholds[2]} XP) | Смертельно ({party_thresholds[3]} XP)")
    
    # Сравниваем XP с порогами для группы
    if monster_xp < party_thresholds[0]:
        st.success("🟢 **Относительная угроза: НИЧТОЖНАЯ**")
        st.write("Для вашей группы этот монстр — просто муха. Они уничтожат его за один раунд, даже не потратив ресурсов.")
    elif monster_xp < party_thresholds[1]:
        st.success("🟢 **Относительная угроза: ЛЕГКАЯ**")
        st.write("Хорошая разминка. Игроки могут потерять немного здоровья, но без риска для жизни.")
    elif monster_xp < party_thresholds[2]:
        st.info("🟡 **Относительная угроза: СРЕДНЯЯ**")
        st.write("Потребует от группы тактики и траты ячеек заклинаний. Возможны сильные ранения.")
    elif monster_xp < party_thresholds[3]:
        st.warning("🟠 **Относительная угроза: СЛОЖНАЯ**")
        st.write("Кто-то из игроков почти наверняка упадет без сознания (0 HP). Нужна хорошая командная работа.")
    else:
        st.error("🔴 **Относительная угроза: СМЕРТЕЛЬНАЯ (TPK)**")
        st.write("Этот монстр способен уничтожить всю партию (Total Party Kill). Бегите, глупцы!")