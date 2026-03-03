import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import joblib
from datetime import datetime

# Function to fetch weather data from Open-Meteo
def fetch_weather_data(lat=50.45, lon=30.52, start_date='2023-01-01', end_date='2026-12-31'):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum"
        f"&timezone=Europe%2FKyiv"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.text}")
    data = response.json()
    daily_data = data['daily']
    df = pd.DataFrame(daily_data)
    df['time'] = pd.to_datetime(df['time'])
    return df

# Streamlit app
st.title("Прогноз опадів на основі даних Open-Meteo")

# Section 1: Load data
st.header("1. Завантаження даних")
option = st.radio("Оберіть спосіб завантаження даних:", ("Завантажити CSV", "Отримати з Open-Meteo"))

data_df = None
if option == "Завантажити CSV":
    uploaded_file = st.file_uploader("Завантажте CSV файл", type="csv")
    if uploaded_file is not None:
        data_df = pd.read_csv(uploaded_file, parse_dates=['time'])
        st.success("Дані завантажено з CSV!")
elif option == "Отримати з Open-Meteo":
    lat = st.number_input("Широта (latitude)", value=50.45)
    lon = st.number_input("Довгота (longitude)", value=30.52)
    start_date = st.date_input("Дата початку", value=datetime(2023, 1, 1))
    end_date = st.date_input("Дата кінця", value=datetime(2023, 12, 31))
    if st.button("Отримати дані з Open-Meteo"):
        try:
            data_df = fetch_weather_data(lat, lon, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            data_df.to_csv("weather_daily.csv", index=False)
            st.success("Дані отримано та збережено в weather_daily.csv!")
        except Exception as e:
            st.error(str(e))

if data_df is not None:
    st.dataframe(data_df.head())

# Section 2: Train model
st.header("2. Навчання моделі")
if data_df is not None or st.button("Навчити модель з weather_daily.csv"):
    if data_df is None:
        try:
            data_df = pd.read_csv("weather_daily.csv", parse_dates=['time'])
        except FileNotFoundError:
            st.error("Файл weather_daily.csv не знайдено. Спочатку завантажте дані.")
            data_df = None

    if data_df is not None:
        # Prepare data
        df = data_df.copy()
        df['temp_max_lag1'] = df['temperature_2m_max'].shift(1)
        df['temp_min_lag1'] = df['temperature_2m_min'].shift(1)
        df['rain_sum_lag1'] = df['rain_sum'].shift(1)
        df['precip'] = (df['precipitation_sum'] > 0).astype(int)
        df = df.dropna()

        X = df[['temp_max_lag1', 'temp_min_lag1', 'rain_sum_lag1']]
        y = df['precip']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        st.write(f"Точність: {acc:.2f}")
        st.text("Звіт класифікації:")
        st.text(report)

        # Save model
        joblib.dump(model, 'model.pkl')
        st.success("Модель навчено та збережено!")

# Section 3: Make prediction
st.header("3. Прогноз опадів")
if st.button("Зробити прогноз"):
    try:
        model = joblib.load('model.pkl')
        data_df = pd.read_csv("weather_daily.csv", parse_dates=['time'])
    except FileNotFoundError:
        st.error("Модель або дані не знайдено. Спочатку навчіть модель.")
    else:
        # For demonstration, predict for the last day using the previous day's data
        last_idx = -1
        input_data = pd.DataFrame({
            'temp_max_lag1': [data_df['temperature_2m_max'].iloc[last_idx-1]],
            'temp_min_lag1': [data_df['temperature_2m_min'].iloc[last_idx-1]],
            'rain_sum_lag1': [data_df['rain_sum'].iloc[last_idx-1]]
        })
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if pred == 1:
            st.success(f"Очікуються опади з ймовірністю {prob:.2f}")
        else:
            st.success("Опадів не очікується")

# Alternative: Predict for a specific date
st.subheader("Прогноз для обраної дати")
if 'model.pkl' in joblib.os.listdir():
    model = joblib.load('model.pkl')
    data_df = pd.read_csv("weather_daily.csv", parse_dates=['time'])
    dates = data_df['time'].dt.date.unique()
    selected_date = st.selectbox("Оберіть дату:", dates[1:])  # Skip first for lag

    if selected_date:
        selected_row = data_df[data_df['time'].dt.date == selected_date]
        prev_date = data_df.iloc[selected_row.index[0] - 1]
        input_data = pd.DataFrame({
            'temp_max_lag1': [prev_date['temperature_2m_max']],
            'temp_min_lag1': [prev_date['temperature_2m_min']],
            'rain_sum_lag1': [prev_date['rain_sum']]
        })
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        actual = "є" if selected_row['precipitation_sum'].values[0] > 0 else "немає"

        if pred == 1:
            st.write(f"Прогноз: Очікуються опади з ймовірністю {prob:.2f}")
        else:
            st.write("Прогноз: Опадів не очікується")
        st.write(f"Фактичні опади: {actual}")
else:
    st.info("Спочатку навчіть модель.")