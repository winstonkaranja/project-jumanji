from pydantic import BaseModel
import requests
from datetime import datetime

class WeatherData(BaseModel):
    latitude: float
    longitude: float
    forecast_days: int = 3
    past_days: int = 7
    weather: dict = {}

    def fetch(self) -> None:
        """Fetch past, current, and forecast weather from Open-Meteo and save into self.weather."""
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "current_weather": True,
                "past_days": self.past_days,
                "forecast_days": self.forecast_days,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean",
                "timezone": "auto"
            }

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            current = data.get("current_weather", {})
            daily = data.get("daily", {})
            times = daily.get("time", [])

            today = datetime.now().date().isoformat()

            past_summary = []
            forecast_summary = []

            for i, date in enumerate(times):
                day_summary = (
                    f"{date}: Temp {daily['temperature_2m_min'][i]}-{daily['temperature_2m_max'][i]}°C, "
                    f"Humidity {daily['relative_humidity_2m_mean'][i]}%, "
                    f"Rain {daily['precipitation_sum'][i]}mm, "
                    f"Wind Max {daily['wind_speed_10m_max'][i]} km/h"
                )
                if date < today:
                    past_summary.append(day_summary)
                else:
                    forecast_summary.append(day_summary)

            self.weather = {
                "location": f"Lat: {self.latitude}, Lon: {self.longitude}",
                "past_7_days": past_summary[-self.past_days:],
                "current": {
                    "temp": f"{current.get('temperature', 'N/A')}°C",
                    "wind": f"{current.get('windspeed', 'N/A')} km/h"
                },
                "forecast_next_days": forecast_summary[:self.forecast_days]
            }

        except requests.RequestException as e:
            self.weather = {"error": str(e)}
        except Exception as ex:
            self.weather = {"error": str(ex)}

# # Example usage:
# weather_instance = WeatherData(latitude=-1.2921, longitude=36.8219)
# weather_instance.fetch()
# print(weather_instance.weather)

