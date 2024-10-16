# EcowittWeatherBot

EcowittWeatherBot is a Telegram bot that allows users to receive real-time weather data from an Ecowitt weather station. Users can request the current weather data or schedule daily notifications to receive updates automatically.

## ✨ Features

- Retrieve real-time weather data from the Ecowitt API.
- Display outdoor and indoor temperature, humidity, wind, rainfall, solar radiation, UV index, and pressure.
- Schedule daily weather updates at a user-specified time.
- Simple inline menu for easy navigation.

## 🛠️ Installation

Follow these steps to set up EcowittWeatherBot:

### 📋 Prerequisites

- **Python 3.7** or higher
- A **Telegram bot token** (obtainable from [BotFather](https://t.me/BotFather))
- Access to the **Ecowitt API** (`application_key`, `api_key`, and device MAC address)

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

4. **Configure the API URL and Credentials**

   The bot uses the Ecowitt API to fetch weather data. Update the `bot.py` file with your actual details:

   - **Replace the placeholders with your actual values:**

     ```python
     API_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
     APPLICATION_KEY = 'YOUR_ECOWITT_APPLICATION_KEY'
     API_KEY = 'YOUR_ECOWITT_API_KEY'
     MAC_ADDRESS = 'YOUR_ECOWITT_MAC_ADDRESS'
     ```

   - **Update the `url` variable in `bot.py`:**

     ```python
     url = (
         f'https://api.ecowitt.net/api/v3/device/real_time'
         f'?application_key={APPLICATION_KEY}'
         f'&api_key={API_KEY}'
         f'&mac={MAC_ADDRESS}'
         f'&call_back=all'
         f'&temp_unitid=1'
         f'&pressure_unitid=5'
         f'&wind_speed_unitid=7'
         f'&rainfall_unitid=12'
         f'&solar_irradiance_unitid=16'
     )
     ```

   **Note:** Replace the following placeholders:

   - `YOUR_TELEGRAM_BOT_TOKEN`: Your Telegram bot token.
   - `YOUR_ECOWITT_APPLICATION_KEY`: Your Ecowitt application key.
   - `YOUR_ECOWITT_API_KEY`: Your Ecowitt API key.
   - `YOUR_ECOWITT_MAC_ADDRESS`: The MAC address of your Ecowitt device.

   **Example:**

   ```python
   API_TOKEN = '1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZ'
   APPLICATION_KEY = 'E6067D110ED009F99AA7864DAF8645EF'
   API_KEY = '5fe46b54-ed45-4665-be61-8d7b474474d5'
   MAC_ADDRESS = '88:T4:84:88:F8:DD'
   ```

5. **Run the bot**

   ```bash
   python bot.py
   ```

## ✅ Try It Out!

You can also try out a demo version of the bot here: [EcowittWeatherBot Demo](https://t.me/ecowittweather_bot)

- `/start` - Initiates the bot and displays the main menu.
- Click **"Weather"** to receive the current weather data.
- Click **"Set Time"** to schedule daily weather updates. The bot will prompt you to enter the desired time in the format `HH:MM` (24-hour).
- Click **"Cancel Notifications"** to stop receiving daily updates.

## 🗓️ Scheduling Notifications

The bot uses the `APScheduler` library to allow users to schedule daily notifications for weather updates. To set a time for daily notifications:

1. Click **"Set Time"** from the inline menu.
2. Enter the time in `HH:MM` format (24-hour format). For example, `09:00`.
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

I hope this updated README provides a clear and well-formatted guide for setting up and using the EcowittWeatherBot. Let me know if there's anything else you'd like me to adjust or add!
