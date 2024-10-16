# EcowittWeatherBot

EcowittWeatherBot is a Telegram bot that allows users to receive real-time weather data from an Ecowitt weather station. Users can request the current weather data or schedule daily notifications to receive updates automatically.

## ✨ Features

- Retrieve real-time weather data from the Ecowitt API.
- Display both outdoor and indoor temperature, humidity, wind, rainfall, solar radiation, UV index, and pressure.
- Schedule daily weather updates at a user-specified time.
- Simple inline menu for easy navigation.

## 🛠️ Installation

Follow these steps to set up EcowittWeatherBot:

### 📋 Prerequisites

- Python 3.7 or higher
- A Telegram bot token (obtainable from [BotFather](https://t.me/botfather))
- Access to the Ecowitt API (application_key, api_key, and device MAC address)

### ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/CyberScopeToday/ecowittweatherbot.git
   cd ecowittweatherbot
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the bot**
   Open the `bot.py` file and replace the placeholders with your actual credentials:
   - `API_TOKEN`: Your Telegram bot token.
   - `APPLICATION_KEY`: Ecowitt application key.
   - `API_KEY`: Ecowitt API key.
   - `MAC_ADDRESS`: The MAC address of your Ecowitt device.

   Example:
   ```python
   API_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
   APPLICATION_KEY = 'YOUR_APPLICATION_KEY'
   API_KEY = 'YOUR_API_KEY'
   MAC_ADDRESS = 'YOUR_MAC_ADDRESS'
   ```

5. **Run the bot**
   ```bash
   python bot.py
   ```
## ✅ Test

You can also try out a demo version of the bot here: [EcowittWeatherBot Demo](https://t.me/ecowittweather_bot)

- `/start` - Initiates the bot and displays the main menu.
- Click "Weather" to receive the current weather data.
- Click "Set time" to schedule daily weather updates. The bot will prompt you to enter the desired time in the format `HH:MM` (24-hour).
- Click "Cancel notifications" to stop receiving daily updates.

## 🗓️ Scheduling Notifications

The bot uses the `APScheduler` library to allow users to schedule daily notifications for weather updates. To set a time for daily notifications:

1. Click "Set time" from the inline menu.
2. Enter the time in `HH:MM` format (24-hour format).
3. The bot will save this schedule and send you the weather update every day at the specified time.

## 📊 Example Output

Below is an example of how the bot displays the weather data:

```
🌡️ Outdoor:
- Temperature: 22.5 °C
- Feels like: 21.8 °C
- Dew point: 15.3 °C
- Humidity: 68%

🏠 Indoor:
- Temperature: 24.0 °C
- Humidity: 55%

💨 Wind:
- Speed: 12.3 km/h
- Gusts: 18.0 km/h
- Direction: 270°

🌧️ Rainfall:
- Rain rate: 0 mm/hr
- Today: 5.6 mm
- Month: 25.0 mm
- Year: 450.7 mm

🌞 Sun and UV:
- Solar radiation: 600 W/m²
- UV index: 3.5

🌪️ Pressure:
- Relative: 1015.2 hPa
- Absolute: 1012.8 hPa
```

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## 🤝 Contributing

Feel free to open issues and pull requests to improve this project. Any contributions are greatly appreciated.

## 📞 Contact

For questions or support, reach out to [CyberScopeToday](https://github.com/CyberScopeToday).

---
We hope this README provides a comprehensive overview of the setup and usage of the EcowittWeatherBot. Let me know if there's anything you'd like me to adjust or add!
