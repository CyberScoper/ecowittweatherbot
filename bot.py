import telebot
import requests
import json
import os
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.base import JobLookupError
from datetime import datetime
from telebot import types

API_TOKEN = 'API_TOKEN' 
bot = telebot.TeleBot(API_TOKEN)

USER_DATA_FILE = 'user_data.json'

scheduler = BackgroundScheduler()
scheduler.start()

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    else:
        return {}

def save_user_data(data):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(data, f)

def get_weather_data():
    try:
        url = 'https://api.ecowitt.net/api/v3/device/real_time?application_key=YOUR_APPLICATION_KEY&api_key=YOUR_API_KEY&mac=YOUR_MAC_ADDRESS&call_back=all&temp_unitid=1&pressure_unitid=5&wind_speed_unitid=7&rainfall_unitid=12&solar_irradiance_unitid=16'
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
🌡️ *Outdoor:*
- Temperature: {outdoor['temperature']['value']} {outdoor['temperature']['unit']}
- Feels like: {outdoor['feels_like']['value']} {outdoor['feels_like']['unit']}
- Dew point: {outdoor['dew_point']['value']} {outdoor['dew_point']['unit']}
- Humidity: {outdoor['humidity']['value']}{outdoor['humidity']['unit']}

🏠 *Indoor:*
- Temperature: {indoor['temperature']['value']} {indoor['temperature']['unit']}
- Humidity: {indoor['humidity']['value']}{indoor['humidity']['unit']}

💨 *Wind:*
- Speed: {wind['wind_speed']['value']} {wind['wind_speed']['unit']}
- Gusts: {wind['wind_gust']['value']} {wind['wind_gust']['unit']}
- Direction: {wind['wind_direction']['value']}°

🌧️ *Rainfall:*
- Rain rate: {rainfall['rain_rate']['value']} {rainfall['rain_rate']['unit']}
- Today: {rainfall['daily']['value']} {rainfall['daily']['unit']}
- Month: {rainfall['monthly']['value']} {rainfall['monthly']['unit']}
- Year: {rainfall['yearly']['value']} {rainfall['yearly']['unit']}

🌞 *Sun and UV:*
- Solar radiation: {solar_and_uvi['solar']['value']} {solar_and_uvi['solar']['unit']}
- UV index: {solar_and_uvi['uvi']['value']}

🌪️ *Pressure:*
- Relative: {pressure['relative']['value']} {pressure['relative']['unit']}
- Absolute: {pressure['absolute']['value']} {pressure['absolute']['unit']}
"""
    return message

def send_scheduled_weather(chat_id):
    data = get_weather_data()
    if data:
        weather_message = format_weather_data(data)
        bot.send_message(chat_id, weather_message, parse_mode='Markdown', reply_markup=main_menu_inline())
    else:
        bot.send_message(chat_id, "Sorry, could not retrieve weather data.", reply_markup=main_menu_inline())

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
    for chat_id, user_time in user_data.items():
        schedule_job(chat_id, user_time)

initialize_jobs()

def main_menu_inline():
    markup = types.InlineKeyboardMarkup()
    weather_button = types.InlineKeyboardButton("Weather", callback_data='weather')
    set_time_button = types.InlineKeyboardButton("Set time", callback_data='set_time')
    cancel_button = types.InlineKeyboardButton("Cancel notifications", callback_data='cancel_notifications')
    markup.add(weather_button)
    markup.add(set_time_button)
    markup.add(cancel_button)
    return markup

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(
        message.chat.id,
        "Hello! I am the weather station bot.\n\nChoose an action:",
        reply_markup=main_menu_inline()
    )

@bot.callback_query_handler(func=lambda call: call.data == 'weather')
def callback_weather(call):
    data = get_weather_data()
    if data:
        weather_message = format_weather_data(data)
        if call.message.text != weather_message:
            try:
                bot.edit_message_text(
                    chat_id=call.message.chat.id,
                    message_id=call.message.message_id,
                    text=weather_message,
                    parse_mode='Markdown',
                    reply_markup=main_menu_inline()
                )
            except telebot.apihelper.ApiTelegramException as e:
                if "message is not modified" in str(e):
                    pass
                else:
                    print(f"Ошибка при редактировании сообщения: {e}")
        else:
           
            pass
    else:
        bot.answer_callback_query(
            callback_query_id=call.id,
            text="Не удалось получить данные о погоде.",
            show_alert=True
        )


@bot.callback_query_handler(func=lambda call: call.data == 'set_time')
def callback_set_time(call):
    try:
        bot.edit_message_text(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            text="Please enter the time in HH:MM format (24-hour format). For example, 09:00",
            reply_markup=None
        )
    except telebot.apihelper.ApiTelegramException as e:
        if "message is not modified" in str(e):
            pass
        else:
            print(f"Ошибка при редактировании сообщения: {e}")
    bot.register_next_step_handler(call.message, process_time_step)


def process_time_step(message):
    try:
        user_time = message.text.strip()
        # Validate the time format
        datetime.strptime(user_time, '%H:%M')
        user_data = load_user_data()
        chat_id = str(message.chat.id)
        user_data[chat_id] = user_time
        save_user_data(user_data)
        schedule_job(chat_id, user_time)
        bot.send_message(
            message.chat.id,
            f"Notifications set for {user_time} every day.",
            reply_markup=main_menu_inline()
        )
    except ValueError:
        msg = bot.send_message(
            message.chat.id,
            "Invalid time format. Please enter the time in HH:MM format."
        )
        bot.register_next_step_handler(msg, process_time_step)

@bot.callback_query_handler(func=lambda call: call.data == 'cancel_notifications')
def callback_cancel_notifications(call):
    chat_id = str(call.message.chat.id)
    user_data = load_user_data()
    if chat_id in user_data:
        del user_data[chat_id]
        save_user_data(user_data)
        job_id = f"weather_{chat_id}"
        try:
            scheduler.remove_job(job_id)
        except JobLookupError:
            pass
        new_text = "Ежедневные уведомления отменены."
    else:
        new_text = "У вас нет активных уведомлений."
    if call.message.text != new_text:
        try:
            bot.edit_message_text(
                chat_id=call.message.chat.id,
                message_id=call.message.message_id,
                text=new_text,
                reply_markup=main_menu_inline()
            )
        except telebot.apihelper.ApiTelegramException as e:
            if "message is not modified" in str(e):
                pass
            else:
                print(f"Ошибка при редактировании сообщения: {e}")
    else:
        pass

bot.polling()
