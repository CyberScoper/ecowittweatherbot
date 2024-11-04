import telebot
import requests
import json
import os
import csv
import io
import logging
from io import BytesIO
from tensorflow.keras.models import load_model
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.base import JobLookupError
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, timedelta
from telebot import types
from keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_logs.log"),  # Сохраняет логи в файл
        logging.StreamHandler()  # Опционально: выводит логи в консоль
    ]
)

logger = logging.getLogger(__name__)

API_TOKEN = 'api from papa bot'
bot = telebot.TeleBot(API_TOKEN)

USER_DATA_FILE = 'user_data.json'
WEATHER_DATA_FILE = 'weather_data.csv'  # Имя файла для сохранения данных

scheduler = BackgroundScheduler()
scheduler.start()

print("Начата запись данных о погоде в файл weather_data.csv")

############################################################начало части с прогнозами

def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), :])
        y.append(dataset[i + look_back, :])
    return np.array(X), np.array(y)


def train_lstm_model():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    import os

    try:
        logger.info("Запуск обучения модели LSTM.")
        
        # Загрузка данных
        data = pd.read_csv('weather_data.csv')
        logger.debug(f"Загруженные данные: {data.head()}")  # Логируем пример данных
        logger.info("Данные успешно загружены.")
        
        # Предобработка данных
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        data.ffill(inplace=True)
        data = data.resample('5min').mean()
        data.ffill(inplace=True)
        logger.debug("Предобработка данных завершена.")

        # Нормализация данных
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values)
        logger.debug(f"Размер нормализованных данных: {scaled_data.shape}")
        dataset = scaled_data
        num_features = dataset.shape[1]

        # Создание последовательностей данных
        look_back = 10
        train_size = int(len(dataset) * 0.8)
        train, _ = dataset[0:train_size], dataset[train_size:]
        X_train, y_train = create_dataset(train, look_back)
        logger.debug(f"Размеры X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info("Датасет для обучения подготовлен.")

        # Обучение модели
        model = Sequential()
        model.add(Input(shape=(look_back, num_features)))
        model.add(LSTM(50))
        model.add(Dense(num_features))

        model.compile(loss='mean_squared_error', optimizer='adam')

        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
        logger.info("Обучение модели завершено.")

        # Сохранение модели и scaler
        model.save('lstm_model.keras')
        np.save('scaler.npy', scaler.scale_)
        np.save('scaler_min.npy', scaler.min_)
        logger.info("Модель и scaler успешно сохранены.")

    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}")

def predict_with_lstm():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import load_model

    # Загрузка данных
    data = pd.read_csv('weather_data.csv')

    # Предобработка данных
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    data.ffill(inplace=True)
    data = data.resample('5min').mean()
    data.ffill(inplace=True)

    # Загрузка scaler
    scale_ = np.load('scaler.npy')
    min_ = np.load('scaler_min.npy')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.scale_ = scale_
    scaler.min_ = min_
    scaler.data_min_ = data.values.min(axis=0)
    scaler.data_max_ = data.values.max(axis=0)
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_

    # Нормализация данных
    scaled_data = scaler.transform(data.values)
    dataset = scaled_data
    num_features = dataset.shape[1]

    # Загрузка модели
    model = load_model('lstm_model.keras') #('lstm_model.h5')
    model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mean_absolute_error', 'mean_absolute_percentage_error']  # Добавление MAPE
)

    # Подготовка данных для прогнозирования
    look_back = 10
    last_data = dataset[-look_back:]
    future_predict = []

    for i in range(24):
        last_data_reshaped = np.reshape(last_data, (1, look_back, num_features))
        next_pred = model.predict(last_data_reshaped)
        future_predict.append(next_pred[0])
        last_data = np.vstack([last_data[1:], next_pred])

    # Обратное преобразование прогнозов
    future_predict_inverse = scaler.inverse_transform(np.array(future_predict))

    # Создание индексов для будущих прогнозов
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(minutes=5), periods=24, freq='5min')
    features = ['temperature', 'humidity', 'pressure']
    future_forecast_df = pd.DataFrame(future_predict_inverse, index=future_dates, columns=features)

    return future_forecast_df

train_lstm_model()

from apscheduler.triggers.cron import CronTrigger

scheduler.add_job(
    train_lstm_model,
    trigger=IntervalTrigger(hours=1),  # Trains the model every hour
    id='train_lstm_model',
    replace_existing=True
)

############################################################конец части с прогнозами

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}  # Если файл пустой или поврежден, инициализируем пустой словарь
        # Преобразуем старый формат данных в новый формат
        for chat_id, settings in data.items():
            if isinstance(settings, str):  # Если значение — строка (время)
                data[chat_id] = {
                    'notification_time': settings,
                    'pressure_alert': False,  # По умолчанию отключаем уведомления о давлении
                    'comfort_alert': False,   # По умолчанию отключаем уведомления о комфорте
                    'recommendation_alert': False  # По умолчанию отключаем уведомления рекомендаций
                }
            else:
                # Убедимся, что все необходимые ключи присутствуют
                settings.setdefault('pressure_alert', False)
                settings.setdefault('comfort_alert', False)
                settings.setdefault('recommendation_alert', False)
        return data
    else:
        return {}

def save_user_data(data):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(data, f)

def get_weather_data():
    try:
        url = 'https://api.ecowitt.net/api/v3/device/real_time?application_key=E6065D116ED009F99AA9004DAF8718EF&api_key=7fe45b77-ed01-4679-be61-8d7b961474b0&mac=34:94:54:8C:F8:CA&call_back=all&temp_unitid=1&pressure_unitid=5&wind_speed_unitid=7&rainfall_unitid=12&solar_irradiance_unitid=16'
        response = requests.get(url)
        data = response.json()
        return data
    except Exception as e:
        print(f"Error while getting data: {e}")
        return None

def format_weather_data(data):
    outdoor = data['data']['outdoor']
    indoor = data['data']['indoor']
    wind = data['data']['wind']
    pressure = data['data']['pressure']
    rainfall = data['data']['rainfall']
    solar_and_uvi = data['data']['solar_and_uvi']
    
    message = f"""
🌡️ *Погода на улице*: {outdoor['temperature']['value']} {outdoor['temperature']['unit']}, ощущается как {outdoor['feels_like']['value']} {outdoor['feels_like']['unit']}
- Влажность: {outdoor['humidity']['value']}{outdoor['humidity']['unit']} | Точка росы: {outdoor['dew_point']['value']} {outdoor['dew_point']['unit']}

🏠 *В помещении*: {indoor['temperature']['value']} {indoor['temperature']['unit']} | Влажность: {indoor['humidity']['value']}{indoor['humidity']['unit']}

💨 *Ветер*: {wind['wind_speed']['value']} {wind['wind_speed']['unit']} (порывы до {wind['wind_gust']['value']} {wind['wind_gust']['unit']}), направление: {wind['wind_direction']['value']}°

🌧️ *Осадки*: {rainfall['rain_rate']['value']} {rainfall['rain_rate']['unit']} | Сегодня: {rainfall['daily']['value']} {rainfall['daily']['unit']}, Месяц: {rainfall['monthly']['value']} {rainfall['monthly']['unit']}, Год: {rainfall['yearly']['value']} {rainfall['yearly']['unit']}

🌞 *Солнце и UV*: Радиация {solar_and_uvi['solar']['value']} {solar_and_uvi['solar']['unit']} | UV: {solar_and_uvi['uvi']['value']}

🌪️ *Давление*: Отн. {pressure['relative']['value']} {pressure['relative']['unit']} | Абс. {pressure['absolute']['value']} {pressure['absolute']['unit']}
"""
    return message


def generate_recommendations(data):
    try:
        outdoor = data['data']['outdoor']
        temperature = float(outdoor['temperature']['value'])
        humidity = float(outdoor['humidity']['value'])
        wind = data['data']['wind']
        wind_speed = float(wind['wind_speed']['value'])
        wind_unit = wind['wind_speed']['unit']
        solar_radiation = float(data['data']['solar_and_uvi']['solar']['value'])
        solar_unit = data['data']['solar_and_uvi']['solar']['unit']

        recommendations = ""

        # Советы по одежде
        if temperature < 0:
            recommendations += "🧥 На улице холодно. Рекомендуется надеть теплую одежду и шапку.\n"
        elif 0 <= temperature < 10:
            recommendations += "🧣 Прохладная погода. Одевайтесь теплее и не забудьте шарф.\n"
        elif 10 <= temperature < 20:
            recommendations += "👕 Погода прохладная. Рекомендуется легкая куртка или свитер.\n"
        elif 20 <= temperature < 30:
            recommendations += "👚 Теплая погода. Можно одеться полегче.\n"
        elif temperature >= 30:
            recommendations += "🩳 Жарко! Одевайтесь в легкую и дышащую одежду.\n"

        # Учитываем скорость ветра
        if wind_speed > 10:
            recommendations += f"🌬️ Сильный ветер ({wind_speed} {wind_unit}). Одевайтесь соответственно и будьте осторожны на улице.\n"

        # Рекомендации по активности
        if solar_radiation > 800:
            recommendations += f"🌞 Высокая солнечная активность ({solar_radiation} {solar_unit}). Используйте солнцезащитный крем и избегайте долгого пребывания на солнце.\n"

        if not recommendations:
            recommendations = "✅ Погодные условия благоприятны. Наслаждайтесь вашим днем!"

        return recommendations

    except Exception as e:
        print(f"Ошибка при генерации рекомендаций: {e}")
        return "Извините, не удалось получить рекомендации по одежде и активности."

def save_weather_data():
    data = get_weather_data()
    if data:
        try:
            outdoor = data['data']['outdoor']
            pressure = data['data']['pressure']
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            temperature = outdoor['temperature']['value']
            humidity = outdoor['humidity']['value']
            pressure_value = pressure['relative']['value']

            # Проверяем, существует ли файл, чтобы записать заголовки
            file_exists = os.path.isfile(WEATHER_DATA_FILE)

            with open(WEATHER_DATA_FILE, mode='a', newline='') as csv_file:
                fieldnames = ['timestamp', 'temperature', 'humidity', 'pressure']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow({
                    'timestamp': timestamp,
                    'temperature': temperature,
                    'humidity': humidity,
                    'pressure': pressure_value
                })
            print(f"Данные записаны в файл: {timestamp}, {temperature}, {humidity}, {pressure_value}")
        except Exception as e:
            print(f"Ошибка при сохранении данных: {e}")
    else:
        print("Не удалось получить данные для сохранения.")


def send_scheduled_weather(chat_id):
    data = get_weather_data()
    if data:
        weather_message = format_weather_data(data)
        bot.send_message(chat_id, weather_message, parse_mode='Markdown', reply_markup=main_menu_inline(chat_id))
        # Проверяем экстремальные условия и отправляем предупреждения
        check_and_send_comfort_alert(chat_id, data)
    else:
        bot.send_message(chat_id, "Извините, не удалось получить данные о погоде.", reply_markup=main_menu_inline(chat_id))

def schedule_job(chat_id, user_time):
    job_id = f"weather_{chat_id}"
    try:
        scheduler.remove_job(job_id)
    except JobLookupError:
        pass

    hour, minute = map(int, user_time.split(':'))
    scheduler.add_job(
        send_scheduled_weather,
        'cron',
        args=[chat_id],
        hour=hour,
        minute=minute,
        id=job_id
    )

def initialize_jobs():
    user_data = load_user_data()
    for chat_id, settings in user_data.items():
        if isinstance(settings, dict) and 'notification_time' in settings:
            schedule_job(chat_id, settings['notification_time'])

initialize_jobs()

def main_menu_inline(chat_id):
    user_data = load_user_data()
    chat_id_str = str(chat_id)
    user_settings = user_data.get(chat_id_str, {})
    notification_time = user_settings.get('notification_time')
    pressure_alert = user_settings.get('pressure_alert', False)
    comfort_alert = user_settings.get('comfort_alert', False)
    recommendation_alert = user_settings.get('recommendation_alert', False)

    markup = types.InlineKeyboardMarkup()
    forecast_temp_button = types.InlineKeyboardButton("Прогноз температуры", callback_data='forecast_temperature')
    forecast_humidity_button = types.InlineKeyboardButton("Прогноз влажности", callback_data='forecast_humidity')
    forecast_pressure_button = types.InlineKeyboardButton("Прогноз давления", callback_data='forecast_pressure')

    weather_button = types.InlineKeyboardButton("Погода", callback_data='weather')
    history_button = types.InlineKeyboardButton("История", callback_data='history')
    recommendation_button = types.InlineKeyboardButton("Получить рекомендацию", callback_data='get_recommendation')

    if notification_time:
        set_time_button = types.InlineKeyboardButton("Отменить уведомления", callback_data='toggle_notifications')
    else:
        set_time_button = types.InlineKeyboardButton("Установить время", callback_data='toggle_notifications')

    if pressure_alert:
        pressure_button = types.InlineKeyboardButton("Отписаться от предупреждений о дожде", callback_data='toggle_pressure_alert')
    else:
        pressure_button = types.InlineKeyboardButton("Подписаться на предупреждения о дожде", callback_data='toggle_pressure_alert')

    if comfort_alert:
        comfort_button = types.InlineKeyboardButton("Отписаться от предупреждений о комфорте", callback_data='toggle_comfort_alert')
    else:
        comfort_button = types.InlineKeyboardButton("Подписаться на предупреждения о комфорте", callback_data='toggle_comfort_alert')

    if recommendation_alert:
        recommendation_alert_button = types.InlineKeyboardButton("Отписаться от рекомендаций", callback_data='toggle_recommendation_alert')
    else:
        recommendation_alert_button = types.InlineKeyboardButton("Подписаться на рекомендации", callback_data='toggle_recommendation_alert')

    # Располагаем кнопки
    markup.add(weather_button, history_button)
    markup.add(recommendation_button)
    markup.add(set_time_button)
    markup.add(pressure_button)
    markup.add(comfort_button)
    markup.add(recommendation_alert_button)
    markup.add(forecast_temp_button, forecast_humidity_button, forecast_pressure_button)

    return markup

@bot.callback_query_handler(func=lambda call: call.data == 'forecast_temperature')
def callback_forecast_temperature(call):
    user = call.from_user
    print(f"User {user.first_name} (@{user.username}) requested temperature forecast.")
    try:
        # Удаляем предыдущее сообщение
        bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
        # Получаем прогноз
        forecast_df = predict_with_lstm()
        # Строим график
        plt.figure(figsize=(12, 4))
        plt.plot(forecast_df.index, forecast_df['temperature'], label='Прогноз температуры', color='orange')
        plt.title('Прогноз температуры на 2 часа')
        plt.xlabel('Время')
        plt.ylabel('Температура')
        plt.legend()
        # Сохраняем график в буфер
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        # Отправляем график пользователю
        bot.send_photo(call.message.chat.id, buf, reply_markup=main_menu_inline(call.message.chat.id))
        buf.close()
    except Exception as e:
        print(f"Ошибка при отправке прогноза температуры: {e}")
        bot.send_message(call.message.chat.id, "Извините, не удалось получить прогноз температуры.", reply_markup=main_menu_inline(call.message.chat.id))

@bot.callback_query_handler(func=lambda call: call.data == 'forecast_humidity')
def callback_forecast_humidity(call):
    user = call.from_user
    print(f"User {user.first_name} (@{user.username}) requested humidity forecast.")
    try:
        bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
        forecast_df = predict_with_lstm()
        plt.figure(figsize=(12, 4))
        plt.plot(forecast_df.index, forecast_df['humidity'], label='Прогноз влажности', color='blue')
        plt.title('Прогноз влажности на 2 часа')
        plt.xlabel('Время')
        plt.ylabel('Влажность')
        plt.legend()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        bot.send_photo(call.message.chat.id, buf, reply_markup=main_menu_inline(call.message.chat.id))
        buf.close()
    except Exception as e:
        print(f"Ошибка при отправке прогноза влажности: {e}")
        bot.send_message(call.message.chat.id, "Извините, не удалось получить прогноз влажности.", reply_markup=main_menu_inline(call.message.chat.id))

@bot.callback_query_handler(func=lambda call: call.data == 'forecast_pressure')
def callback_forecast_pressure(call):
    user = call.from_user
    print(f"User {user.first_name} (@{user.username}) requested pressure forecast.")
    try:
        bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
        forecast_df = predict_with_lstm()
        plt.figure(figsize=(12, 4))
        plt.plot(forecast_df.index, forecast_df['pressure'], label='Прогноз давления', color='green')
        plt.title('Прогноз давления на 2 часа')
        plt.xlabel('Время')
        plt.ylabel('Давление')
        plt.legend()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        bot.send_photo(call.message.chat.id, buf, reply_markup=main_menu_inline(call.message.chat.id))
        buf.close()
    except Exception as e:
        print(f"Ошибка при отправке прогноза давления: {e}")
        bot.send_message(call.message.chat.id, "Извините, не удалось получить прогноз давления.", reply_markup=main_menu_inline(call.message.chat.id))

@bot.callback_query_handler(func=lambda call: call.data == 'get_recommendation')
def callback_get_recommendation(call):
    data = get_weather_data()
    if data:
        recommendations = generate_recommendations(data)
        try:
            # Логируем взаимодействие пользователя
            user = call.from_user
            print(f"User {user.first_name} (@{user.username}) requested recommendations.")
            # Удаляем предыдущее сообщение
            bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
            # Отправляем рекомендации с обновленной клавиатурой
            bot.send_message(
                chat_id=call.message.chat.id,
                text=recommendations,
                reply_markup=main_menu_inline(call.message.chat.id)
            )
        except telebot.apihelper.ApiTelegramException as e:
            print(f"Ошибка при обработке сообщения: {e}")
            bot.send_message(
                chat_id=call.message.chat.id,
                text="Извините, не удалось получить рекомендации.",
                reply_markup=main_menu_inline(call.message.chat.id)
            )
    else:
        bot.send_message(
            chat_id=call.message.chat.id,
            text="Извините, не удалось получить данные о погоде.",
            reply_markup=main_menu_inline(call.message.chat.id)
        )
@bot.callback_query_handler(func=lambda call: call.data == 'toggle_recommendation_alert')
def callback_toggle_recommendation_alert(call):
    chat_id = str(call.message.chat.id)
    user_data = load_user_data()
    if chat_id not in user_data or not isinstance(user_data[chat_id], dict):
        user_data[chat_id] = {}
    recommendation_alert = user_data[chat_id].get('recommendation_alert', False)

    if recommendation_alert:
        # Currently subscribed, unsubscribe
        user_data[chat_id]['recommendation_alert'] = False
        save_user_data(user_data)
        new_text = "Вы отписались от уведомлений рекомендаций."
    else:
        # Currently unsubscribed, subscribe
        user_data[chat_id]['recommendation_alert'] = True
        save_user_data(user_data)
        new_text = "Вы подписались на уведомления рекомендаций."
    try:
        # Логируем взаимодействие пользователя
        user = call.from_user
        action = "подписался на" if user_data[chat_id]['recommendation_alert'] else "отписался от"
        print(f"User {user.first_name} (@{user.username}) {action} уведомления рекомендаций.")
        # Удаляем предыдущее сообщение
        bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
        # Отправляем новое сообщение с результатом и обновленной клавиатурой
        bot.send_message(
            chat_id=call.message.chat.id,
            text=new_text,
            reply_markup=main_menu_inline(call.message.chat.id)
        )
    except telebot.apihelper.ApiTelegramException as e:
        print(f"Ошибка при обработке сообщения: {e}")

@bot.callback_query_handler(func=lambda call: call.data == 'toggle_comfort_alert')
def callback_toggle_comfort_alert(call):
    chat_id = str(call.message.chat.id)
    user_data = load_user_data()
    if chat_id not in user_data or not isinstance(user_data[chat_id], dict):
        user_data[chat_id] = {}
    comfort_alert = user_data[chat_id].get('comfort_alert', False)

    if comfort_alert:
        # Currently subscribed, unsubscribe
        user_data[chat_id]['comfort_alert'] = False
        save_user_data(user_data)
        new_text = "Вы отписались от предупреждений о комфорте."
    else:
        # Currently unsubscribed, subscribe
        user_data[chat_id]['comfort_alert'] = True
        save_user_data(user_data)
        new_text = "Вы подписались на предупреждения о комфорте."
    try:
        # Логируем взаимодействие пользователя
        user = call.from_user
        action = "подписался на" if user_data[chat_id]['comfort_alert'] else "отписался от"
        print(f"User {user.first_name} (@{user.username}) {action} предупреждения о комфорте.")
        # Удаляем предыдущее сообщение
        bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
        # Отправляем новое сообщение с результатом и обновленной клавиатурой
        bot.send_message(
            chat_id=call.message.chat.id,
            text=new_text,
            reply_markup=main_menu_inline(call.message.chat.id)
        )
    except telebot.apihelper.ApiTelegramException as e:
        print(f"Ошибка при обработке сообщения: {e}")

@bot.callback_query_handler(func=lambda call: call.data == 'toggle_pressure_alert')
def callback_toggle_pressure_alert(call):
    chat_id = str(call.message.chat.id)
    user_data = load_user_data()
    if chat_id not in user_data or not isinstance(user_data[chat_id], dict):
        user_data[chat_id] = {}
    pressure_alert = user_data[chat_id].get('pressure_alert', False)

    if pressure_alert:
        # Currently subscribed, unsubscribe
        user_data[chat_id]['pressure_alert'] = False
        save_user_data(user_data)
        new_text = "Вы отписались от уведомлений о возможном дожде."
    else:
        # Currently unsubscribed, subscribe
        user_data[chat_id]['pressure_alert'] = True
        save_user_data(user_data)
        new_text = "Вы подписались на уведомления о возможном дожде."
    try:
        # Логируем взаимодействие пользователя
        user = call.from_user
        action = "подписался на" if user_data[chat_id]['pressure_alert'] else "отписался от"
        print(f"User {user.first_name} (@{user.username}) {action} уведомления о давлении.")
        # Удаляем предыдущее сообщение
        bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
        # Отправляем новое сообщение с результатом и обновленной клавиатурой
        bot.send_message(
            chat_id=call.message.chat.id,
            text=new_text,
            reply_markup=main_menu_inline(call.message.chat.id)
        )
    except telebot.apihelper.ApiTelegramException as e:
        print(f"Ошибка при обработке сообщения: {e}")

@bot.callback_query_handler(func=lambda call: call.data == 'toggle_notifications')
def callback_toggle_notifications(call):
    chat_id = str(call.message.chat.id)
    user_data = load_user_data()
    if chat_id not in user_data or not isinstance(user_data[chat_id], dict):
        user_data[chat_id] = {}
    notification_time = user_data[chat_id].get('notification_time')

    if notification_time:
        # Notifications are set, cancel them
        del user_data[chat_id]['notification_time']
        save_user_data(user_data)
        job_id = f"weather_{chat_id}"
        try:
            scheduler.remove_job(job_id)
        except JobLookupError:
            pass
        new_text = "Ежедневные уведомления отменены."
        try:
            # Логируем взаимодействие пользователя
            user = call.from_user
            print(f"User {user.first_name} (@{user.username}) отменил ежедневные уведомления.")
            # Удаляем предыдущее сообщение
            bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
            # Отправляем новое сообщение с результатом и обновленной клавиатурой
            bot.send_message(
                chat_id=call.message.chat.id,
                text=new_text,
                reply_markup=main_menu_inline(call.message.chat.id)
            )
        except telebot.apihelper.ApiTelegramException as e:
            print(f"Ошибка при обработке сообщения: {e}")
    else:
        # Notifications are not set, ask user to set time
        try:
            # Логируем взаимодействие пользователя
            user = call.from_user
            print(f"User {user.first_name} (@{user.username}) хочет установить время уведомлений.")
            # Удаляем предыдущее сообщение
            bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
            # Отправляем новое сообщение с запросом времени
            msg = bot.send_message(
                chat_id=call.message.chat.id,
                text="Пожалуйста, введите время в формате HH:MM (24-часовой формат). Например, 09:00",
                reply_markup=None
            )
            bot.register_next_step_handler(msg, process_time_step)
        except telebot.apihelper.ApiTelegramException as e:
            print(f"Ошибка при обработке сообщения: {e}")

@bot.message_handler(commands=['start'])
def send_welcome(message):
    user = message.from_user
    print(f"User {user.first_name} (@{user.username}) started the bot.")
    bot.send_message(
        message.chat.id,
        "Привет! Я бот метеостанции.\n\nВыберите действие:",
        reply_markup=main_menu_inline(message.chat.id)
    )

@bot.callback_query_handler(func=lambda call: call.data == 'weather')
def callback_weather(call):
    user = call.from_user
    print(f"User {user.first_name} (@{user.username}) requested weather.")
    data = get_weather_data()
    if data:
        weather_message = format_weather_data(data)
        try:
            # Удаляем предыдущее сообщение
            bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
            # Отправляем новое сообщение с погодой и клавиатурой
            bot.send_message(
                chat_id=call.message.chat.id,
                text=weather_message,
                parse_mode='Markdown',
                reply_markup=main_menu_inline(call.message.chat.id)
            )
            # Проверяем экстремальные условия и отправляем предупреждения
            check_and_send_comfort_alert(call.message.chat.id, data)
        except telebot.apihelper.ApiTelegramException as e:
            print(f"Ошибка при обработке сообщения: {e}")
            bot.send_message(
                chat_id=call.message.chat.id,
                text="Извините, не удалось получить данные о погоде.",
                reply_markup=main_menu_inline(call.message.chat.id)
            )
    else:
        bot.send_message(
            chat_id=call.message.chat.id,
            text="Извините, не удалось получить данные о погоде.",
            reply_markup=main_menu_inline(call.message.chat.id)
        )

def process_time_step(message):
    try:
        user_time = message.text.strip()
        # Проверяем формат времени
        datetime.strptime(user_time, '%H:%M')
        user_data = load_user_data()
        chat_id = str(message.chat.id)
        if chat_id not in user_data or not isinstance(user_data[chat_id], dict):
            user_data[chat_id] = {}
        user_data[chat_id]['notification_time'] = user_time
        save_user_data(user_data)
        schedule_job(chat_id, user_time)
        # Логируем взаимодействие пользователя
        user = message.from_user
        print(f"User {user.first_name} (@{user.username}) установил время уведомлений на {user_time}.")
        bot.send_message(
            message.chat.id,
            f"Уведомления установлены на {user_time} каждый день.",
            reply_markup=main_menu_inline(message.chat.id)
        )
    except ValueError:
        msg = bot.send_message(
            message.chat.id,
            "Неверный формат времени. Пожалуйста, введите время в формате HH:MM."
        )
        bot.register_next_step_handler(msg, process_time_step)

@bot.callback_query_handler(func=lambda call: call.data == 'history')
def callback_history(call):
    user = call.from_user
    print(f"User {user.first_name} (@{user.username}) requested history.")
    data = get_history_data()
    if data:
        try:
            # Удаляем предыдущее сообщение
            bot.delete_message(chat_id=call.message.chat.id, message_id=call.message.message_id)
            # Генерируем и отправляем график с клавиатурой
            generate_and_send_history_graph(call.message.chat.id, data, reply_markup=main_menu_inline(call.message.chat.id))
        except Exception as e:
            print(f"Error while processing history data: {e}")
            bot.send_message(call.message.chat.id, "Извините, не удалось обработать исторические данные.", reply_markup=main_menu_inline(call.message.chat.id))
    else:
        bot.send_message(call.message.chat.id, "Извините, не удалось получить исторические данные.", reply_markup=main_menu_inline(call.message.chat.id))

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

def get_history_data():
    try:
        application_key = 'E6065D116ED009F99AA9004DAF8718EF'
        api_key = '7fe45b77-ed01-4679-be61-8d7b961474b0'
        mac = '34:94:54:8C:F8:CA'
        temp_unitid = '1'
        pressure_unitid = '5'
        wind_speed_unitid = '7'
        rainfall_unitid = '12'
        solar_irradiance_unitid = '16'

        # Set start_date and end_date with time included
        start_date = datetime.now().strftime('%Y-%m-%d 00:00:00')
        end_date = datetime.now().strftime('%Y-%m-%d 23:59:59')

        # Use a valid cycle_type
        cycle_type = '5min'  # Or 'auto', '30min', '4hour', '1day'
        call_back = 'outdoor.temperature,outdoor.humidity,pressure.relative' # Specify required data

        url = 'https://api.ecowitt.net/api/v3/device/history'

        params = {
            'application_key': application_key,
            'api_key': api_key,
            'mac': mac,
            'start_date': start_date,
            'end_date': end_date,
            'cycle_type': cycle_type,
            'call_back': call_back,
            'temp_unitid': temp_unitid,
            'pressure_unitid': pressure_unitid,
            'wind_speed_unitid': wind_speed_unitid,
            'rainfall_unitid': rainfall_unitid,
            'solar_irradiance_unitid': solar_irradiance_unitid,
        }

        response = requests.get(url, params=params)
        data = response.json()
        return data
    except Exception as e:
        print(f"Error while getting history data: {e}")
        return None

def generate_and_send_history_graph(chat_id, data, reply_markup=None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from io import BytesIO
        import numpy as np
        import seaborn as sns

        # Проверяем наличие необходимых данных
        if ('data' not in data or
            'outdoor' not in data['data'] or
            'temperature' not in data['data']['outdoor'] or
            'humidity' not in data['data']['outdoor'] or
            'pressure' not in data['data'] or
            'relative' not in data['data']['pressure'] or
            'list' not in data['data']['outdoor']['temperature'] or
            'list' not in data['data']['outdoor']['humidity'] or
            'list' not in data['data']['pressure']['relative']):
            bot.send_message(chat_id, "Нет доступных исторических данных.", reply_markup=reply_markup)
            return

        # Получаем данные о температуре
        temp_data = data['data']['outdoor']['temperature']
        temp_unit = temp_data['unit']
        temp_list = temp_data['list']

        # Получаем данные о влажности
        humidity_data = data['data']['outdoor']['humidity']
        humidity_unit = humidity_data['unit']
        humidity_list = humidity_data['list']

        # Получаем данные о давлении
        pressure_data = data['data']['pressure']['relative']
        pressure_unit = pressure_data['unit']
        pressure_list = pressure_data['list']

        # Инициализируем списки
        timestamps = []
        temperatures = []
        humidities = []
        pressures = []

        # Проходим по временным меткам и собираем данные
        for timestamp_str in temp_list.keys():
            timestamp = datetime.fromtimestamp(int(timestamp_str))

            # Температура
            temperature = float(temp_list[timestamp_str])

            # Влажность
            if timestamp_str in humidity_list:
                humidity = float(humidity_list[timestamp_str])
            else:
                humidity = np.nan  # Используем np.nan для отсутствующих значений

            # Давление
            if timestamp_str in pressure_list:
                pressure = float(pressure_list[timestamp_str])
            else:
                pressure = np.nan  # Используем np.nan для отсутствующих значений

            timestamps.append(timestamp)
            temperatures.append(temperature)
            humidities.append(humidity)
            pressures.append(pressure)

        # Сортируем данные по времени
        sorted_data = sorted(zip(timestamps, temperatures, humidities, pressures))
        timestamps, temperatures, humidities, pressures = zip(*sorted_data)

        # Устанавливаем стиль Seaborn
        sns.set_theme(style="darkgrid")

        # Создаем фигуру и подграфики
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # График температуры
        sns.lineplot(x=timestamps, y=temperatures, ax=axes[0], color='tab:red')
        axes[0].set_ylabel(f"Температура ({temp_unit})", fontsize=12, color='tab:red')
        axes[0].tick_params(axis='y', labelcolor='tab:red')
        axes[0].set_title('Температура', fontsize=14)
        axes[0].grid(True)

        # График влажности
        sns.lineplot(x=timestamps, y=humidities, ax=axes[1], color='tab:blue')
        axes[1].set_ylabel(f"Влажность ({humidity_unit})", fontsize=12, color='tab:blue')
        axes[1].tick_params(axis='y', labelcolor='tab:blue')
        axes[1].set_title('Влажность', fontsize=14)
        axes[1].grid(True)

        # График давления
        sns.lineplot(x=timestamps, y=pressures, ax=axes[2], color='tab:green')
        axes[2].set_ylabel(f"Давление ({pressure_unit})", fontsize=12, color='tab:green')
        axes[2].tick_params(axis='y', labelcolor='tab:green')
        axes[2].set_title('Давление', fontsize=14)
        axes[2].grid(True)

        # Настройка оси X (времени)
        axes[2].set_xlabel('Время', fontsize=12)
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate()

        # Добавляем общее название для всей фигуры
        fig.suptitle('История показаний погоды', fontsize=16)

        # Уменьшаем расстояние между подграфиками
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Сохраняем график в буфер и отправляем
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        bot.send_photo(chat_id, buf, reply_markup=reply_markup)
        buf.close()
    except Exception as e:
        print(f"Error while generating the graph: {e}")
        bot.send_message(chat_id, "Извините, произошла ошибка при создании графика.", reply_markup=reply_markup)

def get_history_data_for_pressure():
    try:
        application_key = 'E6065D116ED009F99AA9004DAF8718EF'
        api_key = '7fe45b77-ed01-4679-be61-8d7b961474b0'
        mac = '34:94:54:8C:F8:CA'
        pressure_unitid = '5'

        # Получаем данные за последние 3 часа
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=3)
        start_date = start_time.strftime('%Y-%m-%d %H:%M:%S')
        end_date = end_time.strftime('%Y-%m-%d %H:%M:%S')

        cycle_type = '5min'  # Или 'auto', '30min', в зависимости от доступности данных
        call_back = 'pressure.relative'  # Запрашиваем только данные о давлении

        url = 'https://api.ecowitt.net/api/v3/device/history'

        params = {
            'application_key': application_key,
            'api_key': api_key,
            'mac': mac,
            'start_date': start_date,
            'end_date': end_date,
            'cycle_type': cycle_type,
            'call_back': call_back,
            'pressure_unitid': pressure_unitid,
        }

        response = requests.get(url, params=params)
        data = response.json()
        if data.get('code') == 0:
            return data
        else:
            print(f"API Error: {data.get('msg')}")
            return None
    except Exception as e:
        print(f"Error while getting pressure history data: {e}")
        return None

def calculate_pressure_change(data):
    try:
        # Проверяем наличие необходимых данных
        if ('data' not in data or
            'pressure' not in data['data'] or
            'relative' not in data['data']['pressure'] or
            'list' not in data['data']['pressure']['relative']):
            print("Недостаточно данных для анализа давления.")
            return None

        pressure_data = data['data']['pressure']['relative']
        pressure_unit = pressure_data['unit']
        pressure_list = pressure_data['list']

        # Инициализируем списки
        timestamps = []
        pressures = []

        # Проходим по временным меткам и собираем данные
        for timestamp_str, pressure_value in pressure_list.items():
            timestamp = datetime.fromtimestamp(int(timestamp_str))
            pressure = float(pressure_value)
            timestamps.append(timestamp)
            pressures.append(pressure)

        # Сортируем данные по времени
        sorted_data = sorted(zip(timestamps, pressures))
        timestamps, pressures = zip(*sorted_data)

        # Вычисляем изменение давления
        pressure_change = pressures[-1] - pressures[0]
        return pressure_change
    except Exception as e:
        print(f"Error while calculating pressure change: {e}")
        return None

def analyze_pressure_trend():
    data = get_history_data_for_pressure()
    if data:
        pressure_change = calculate_pressure_change(data)
        if pressure_change is not None:
            # Задаем пороговое значение для падения давления
            threshold = -1.5  # Например, -1.5 мм рт. ст. за 3 часа
            if pressure_change <= threshold:
                # Отправляем уведомление подписанным пользователям
                user_data = load_user_data()
                for chat_id, settings in user_data.items():
                    if isinstance(settings, dict) and settings.get('pressure_alert', False):
                        bot.send_message(
                            chat_id=int(chat_id),
                            text="⚠️ Внимание! Обнаружено значительное падение атмосферного давления. Возможен дождь в ближайшее время."
                        )
        else:
            print("Не удалось вычислить изменение давления.")
    else:
        print("Не удалось получить данные для анализа давления.")

def calculate_heat_index(temperature_c, humidity):
    # Переводим температуру в градусы Фаренгейта
    temperature_f = temperature_c * 9/5 + 32
    if temperature_f >= 80:
        # Формула Heat Index
        hi = -42.379 + 2.04901523*temperature_f + 10.14333127*humidity \
             - 0.22475541*temperature_f*humidity - 0.00683783*temperature_f**2 \
             - 0.05481717*humidity**2 + 0.00122874*temperature_f**2*humidity \
             + 0.00085282*temperature_f*humidity**2 - 0.00000199*temperature_f**2*humidity**2
        # Возвращаем температуру в градусах Цельсия
        heat_index_c = (hi - 32) * 5/9
        return heat_index_c
    else:
        return temperature_c  # Если температура ниже 80°F, Heat Index не рассчитывается

def calculate_wind_chill(temperature_c, wind_speed_kmh):
    # Wind Chill имеет смысл только при температуре ниже 10°C и скорости ветра выше 4.8 км/ч
    if temperature_c <= 10 and wind_speed_kmh > 4.8:
        wc = 13.12 + 0.6215*temperature_c - 11.37*(wind_speed_kmh**0.16) + 0.3965*temperature_c*(wind_speed_kmh**0.16)
        return wc
    else:
        return temperature_c  # Если условия не соблюдаются, возвращаем исходную температуру

def check_and_send_comfort_alert(chat_id, data):
    user_data = load_user_data()
    chat_id_str = str(chat_id)
    user_settings = user_data.get(chat_id_str, {})
    comfort_alert = user_settings.get('comfort_alert', False)

    if not comfort_alert:
        return  # Пользователь не подписан на предупреждения о комфорте

    try:
        outdoor = data['data']['outdoor']
        temperature = float(outdoor['temperature']['value'])
        humidity = float(outdoor['humidity']['value'])
        wind = data['data']['wind']
        wind_speed = float(wind['wind_speed']['value'])

        heat_index = calculate_heat_index(temperature, humidity)
        wind_chill = calculate_wind_chill(temperature, wind_speed)

        alert_message = ""
        if heat_index > 32:  # Порог для опасной жары (например, 32°C)
            alert_message += f"🔥 Внимание! Высокий индекс жары: {heat_index:.1f}°C. Будьте осторожны и избегайте перегрева.\n"
        if wind_chill < 5:  # Порог для экстремального холода (например, -10°C)
            alert_message += f"❄️ Внимание! Низкий индекс охлаждения ветром: {wind_chill:.1f}°C. Одевайтесь теплее.\n"

        if alert_message:
            bot.send_message(chat_id, alert_message)
    except Exception as e:
        print(f"Ошибка при проверке индекса комфорта: {e}")

# Добавляем задачу для периодической проверки индекса комфорта
def analyze_comfort_index():
    data = get_weather_data()
    if data:
        user_data = load_user_data()
        for chat_id, settings in user_data.items():
            if isinstance(settings, dict) and settings.get('comfort_alert', False):
                check_and_send_comfort_alert(int(chat_id), data)
    else:
        print("Не удалось получить данные для анализа индекса комфорта.")

def send_recommendations_to_subscribed_users():
    current_hour = datetime.now().hour
    if 8 <= current_hour < 22:  # Отправляем между 8:00 и 22:00
        data = get_weather_data()
        if data:
            recommendations = generate_recommendations(data)
            user_data = load_user_data()
            for chat_id, settings in user_data.items():
                if isinstance(settings, dict) and settings.get('recommendation_alert', False):
                    try:
                        bot.send_message(
                            chat_id=int(chat_id),
                            text=recommendations
                        )
                        # Логируем отправку рекомендаций
                        print(f"Sent recommendations to user {chat_id}.")
                    except Exception as e:
                        print(f"Ошибка при отправке рекомендаций пользователю {chat_id}: {e}")
        else:
            print("Не удалось получить данные для отправки рекомендаций.")
    else:
        print("Не время для отправки рекомендаций (ночное время).")

scheduler.add_job(
    analyze_pressure_trend,
    trigger=IntervalTrigger(minutes=30),
    id='pressure_analysis',
    replace_existing=True
)

scheduler.add_job(
    analyze_comfort_index,
    trigger=IntervalTrigger(minutes=30),
    id='comfort_analysis',
    replace_existing=True
)

scheduler.add_job(
    send_recommendations_to_subscribed_users,
    trigger=IntervalTrigger(hours=1),
    id='recommendation_notifications',
    replace_existing=True
)

save_weather_data()

scheduler.add_job(
    save_weather_data,
    trigger=IntervalTrigger(minutes=5),  #нужный интервал
    id='save_weather_data',
    replace_existing=True
)

print("Бот запущен и готов к работе.")

bot.infinity_polling(timeout=60, long_polling_timeout=60)
