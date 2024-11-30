# 🌦️ Telegram Weather Station Bot

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/a99aa965e2dc4688b0e28fbfa34bd04c)](https://app.codacy.com/gh/CyberScopeToday/ecowittweatherbot?utm_source=github.com&utm_medium=referral&utm_content=CyberScopeToday/ecowittweatherbot&utm_campaign=Badge_Grade)

This is a Telegram bot that interfaces with a weather station to provide users with real-time weather data, historical weather trends, forecasts using LSTM neural networks, and personalized weather-related notifications and recommendations.

## Features

- **🌤️ Real-Time Weather Data**: Get the latest weather information including temperature, humidity, wind speed, pressure, and more.
- **📈 Historical Data Visualization**: View historical weather data in the form of interactive graphs.
- **🔮 Weather Forecasts**: Receive forecasts for temperature, humidity, and pressure based on LSTM neural network models trained on collected data.
- **⏰ Scheduled Notifications**: Set daily notifications to receive weather updates at a specified time.
- **⚠️ Alerts and Recommendations**:
  - **🌧️ Pressure Alerts**: Subscribe to receive alerts about significant drops in atmospheric pressure, indicating possible rain.
  - **🔥 Comfort Alerts**: Receive notifications when conditions like heat index or wind chill reach extreme levels.
  - **💡 Recommendations**: Get personalized advice on clothing and outdoor activities based on current weather conditions.
- **🤖 Interactive Commands**: Use inline buttons to interact with the bot and access its features easily.

## Setup Instructions

### Prerequisites

- 🐍 Python 3.7 or higher
- 📱 A Telegram account
- 🤖 A registered Telegram bot with a valid **Bot Token**
- ☁️ Weather API access with necessary **API Keys**

## Demo 🔥🔥🔥

🌐 **Try it out**: You can test the bot's features and see it in action by visiting [this Telegram bot](https://t.me/ecowittweather_bot). 

---

### Installation

1. **📥 Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/telegram-weather-station-bot.git
   cd telegram-weather-station-bot
   ```

2. **📦 Install Required Packages**:

   Install the required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. **⚙️ Configuration**:

   - **🔑 Telegram Bot Token**:

     Obtain a bot token from [BotFather](https://core.telegram.org/bots#6-botfather). Replace the placeholder in the code with your bot's API token:

     ```python
     API_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
     ```

   - **🔑 Weather API Keys**:

     Obtain the necessary API keys from your weather data provider. Replace the placeholders in the code with your API keys:

     ```python
     # Replace these values with your actual API keys and parameters
     application_key = 'YOUR_APPLICATION_KEY'
     api_key = 'YOUR_API_KEY'
     mac = 'YOUR_DEVICE_MAC_ADDRESS'
     ```

     **Note**: Be sure to keep your API keys secure and do not share them publicly.

4. **💾 Set Up Data Storage**:

   Ensure that the script has write permissions to the directory where it is running, as it will create and update the following files:

   - `user_data.json`: Stores user preferences and settings.
   - `weather_data.csv`: Logs weather data periodically for training the model.

## Usage Instructions

### 🚀 Running the Bot

Run the bot script using the following command:

```bash
python bot_script.py
```

Replace `bot_script.py` with the actual name of your bot script file.

### 💬 Interacting with the Bot

Start a conversation with your bot on Telegram by searching for its username.

#### Available Commands and Features

- **/start**: Initialize the bot and display the main menu.

- **Main Menu Options**:

  - **🌤️ Погода**: Get the current weather data.

  - **📈 История**: View historical weather data in graphical form.

  - **💡 Получить рекомендацию**: Receive personalized recommendations based on current weather conditions.

  - **⏰ Установить время**: Set a daily notification time to receive weather updates.

  - **🌧️ Подписаться на предупреждения о давлении / Отписаться от предупреждений о давлении**: Subscribe or unsubscribe from pressure alerts indicating possible rain.

  - **🔥 Подписаться на предупреждения о комфорте / Отписаться от предупреждений о комфорте**: Subscribe or unsubscribe from comfort alerts for extreme heat index or wind chill conditions.

  - **💡 Подписаться на рекомендации / Отписаться от рекомендаций**: Subscribe or unsubscribe from hourly recommendations on clothing and activities.

  - **🌡️ Прогноз температуры**: View the temperature forecast for the next two hours.

  - **💧 Прогноз влажности**: View the humidity forecast for the next two hours.

  - **🌪️ Прогноз давления**: View the pressure forecast for the next two hours.

### Features in Detail

#### 🌤️ Real-Time Weather Data

Select **Погода** from the main menu to receive the latest weather data from your weather station, including:

- 🌡️ **Outdoor Temperature**: Current temperature and feels-like temperature.
- 💧 **Outdoor Humidity**: Current humidity and dew point.
- 🏠 **Indoor Conditions**: Indoor temperature and humidity.
- 💨 **Wind**: Wind speed, gusts, and direction.
- 🌧️ **Rainfall**: Rainfall rates and totals.
- ☀️ **Solar Radiation and UV Index**: Solar radiation levels and UV index.
- 🌪️ **Pressure**: Relative and absolute atmospheric pressure.

#### 📈 Historical Data Visualization

Select **История** to receive graphs displaying historical data for temperature, humidity, and pressure collected throughout the current day.

#### 🔮 Weather Forecasts

Use the **Прогноз температуры**, **Прогноз влажности**, and **Прогноз давления** options to receive forecasts for the next two hours, generated by an LSTM neural network model trained on collected data.

#### ⏰ Scheduled Notifications

Set daily notifications by selecting **Установить время** and entering the desired time in HH:MM format. You will receive daily weather updates at this time.

To cancel daily notifications, select **Отменить уведомления** from the main menu.

#### ⚠️ Alerts and Recommendations

- **🌧️ Pressure Alerts**: Subscribe to pressure alerts to be notified of significant drops in atmospheric pressure, which may indicate impending rain.

- **🔥 Comfort Alerts**: Receive alerts when the heat index or wind chill reaches extreme levels, advising you to take appropriate precautions.

- **💡 Recommendations**: Get hourly recommendations on clothing and outdoor activities based on the current weather. To subscribe or unsubscribe, use the respective options in the main menu.

### Data Logging and Model Training

The bot collects weather data every 5 minutes and saves it to `weather_data.csv`. The LSTM model is trained hourly using the collected data. The model and scaler parameters are saved as `lstm_model.keras`, `scaler.npy`, and `scaler_min.npy`.

## Configuration

### 🔑 API Keys and Tokens

Ensure that you have replaced the placeholder values in the script with your actual API keys and tokens.

- **🤖 Telegram Bot Token**: Replace `'YOUR_TELEGRAM_BOT_TOKEN'` with your bot's token obtained from BotFather.

- **☁️ Weather API Keys**: Replace `'YOUR_APPLICATION_KEY'`, `'YOUR_API_KEY'`, and `'YOUR_DEVICE_MAC_ADDRESS'` with your weather station's API keys and MAC address.

### 📝 Logging

The bot logs its activities to the console and to a file named `training_logs.log`. You can monitor this file to see the bot's activities and any errors that may occur.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue if you have any suggestions or find any bugs.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
