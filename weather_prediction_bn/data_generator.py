import numpy as np
import pandas as pd
from bayesian_network import WeatherBayesianNetwork


class WeatherDataGenerator:
    """Generate synthetic weather data for training and testing"""

    def __init__(self):
        self.bn = WeatherBayesianNetwork()

    def generate_dataset(self, n_samples=1000):
        """Generate synthetic dataset from the Bayesian Network"""
        data = []

        for _ in range(n_samples):
            # Sample from the network
            sample = self._sample_from_network()
            data.append(sample)

        df = pd.DataFrame(data)
        return df

    def _sample_from_network(self):
        """Generate a single sample from the network"""
        # Sample Season
        season_idx = np.random.choice(4, p=[0.25, 0.25, 0.25, 0.25])
        season = self.bn.states['Season'][season_idx]

        # Sample Temperature given Season
        temp_probs = {
            'Spring': [0.30, 0.40, 0.30],
            'Summer': [0.10, 0.30, 0.60],
            'Fall': [0.30, 0.40, 0.30],
            'Winter': [0.50, 0.30, 0.20]
        }
        temp_idx = np.random.choice(3, p=temp_probs[season])
        temperature = self.bn.states['Temperature'][temp_idx]

        # Sample Humidity given Season
        hum_probs = {
            'Spring': [0.20, 0.40, 0.40],
            'Summer': [0.30, 0.40, 0.30],
            'Fall': [0.25, 0.35, 0.40],
            'Winter': [0.35, 0.35, 0.30]
        }
        hum_idx = np.random.choice(3, p=hum_probs[season])
        humidity = self.bn.states['Humidity'][hum_idx]

        # Sample Pressure given Season
        press_probs = {
            'Spring': [0.25, 0.50, 0.25],
            'Summer': [0.30, 0.45, 0.25],
            'Fall': [0.25, 0.50, 0.25],
            'Winter': [0.30, 0.40, 0.30]
        }
        press_idx = np.random.choice(3, p=press_probs[season])
        pressure = self.bn.states['Pressure'][press_idx]

        # Get weather probabilities using the network's inference
        evidence = {
            'Temperature': temperature,
            'Humidity': humidity,
            'Pressure': pressure
        }

        # Use the network's prediction method to get probabilities
        weather_probs_dict = self.bn.predict_weather(evidence)
        weather_probs = list(weather_probs_dict.values())

        # Sample weather based on probabilities
        weather_idx = np.random.choice(4, p=weather_probs)
        weather = self.bn.states['Weather'][weather_idx]

        return {
            'Season': season,
            'Temperature': temperature,
            'Humidity': humidity,
            'Pressure': pressure,
            'Weather': weather
        }

    def get_statistics(self, df):
        """Get statistics from generated data"""
        stats = {
            'total_samples': len(df),
            'weather_distribution': df['Weather'].value_counts().to_dict(),
            'season_distribution': df['Season'].value_counts().to_dict(),
            'temperature_distribution': df['Temperature'].value_counts().to_dict(),
            'humidity_distribution': df['Humidity'].value_counts().to_dict(),
            'pressure_distribution': df['Pressure'].value_counts().to_dict()
        }

        # Calculate conditional probabilities from data
        stats['summer_weather'] = df[df['Season'] == 'Summer']['Weather'].value_counts().to_dict()
        stats['rainy_seasons'] = df[df['Weather'] == 'Rainy']['Season'].value_counts().to_dict()

        return stats

    def print_summary(self, df):
        """Print a summary of the generated data"""
        print("\n" + "=" * 50)
        print("DATA GENERATION SUMMARY")
        print("=" * 50)
        print(f"Total samples: {len(df)}")

        print("\nWeather Distribution:")
        weather_counts = df['Weather'].value_counts()
        for weather in self.bn.states['Weather']:
            count = weather_counts.get(weather, 0)
            percentage = (count / len(df)) * 100
            print(f"  {weather}: {count} ({percentage:.1f}%)")

        print("\nSeason Distribution:")
        season_counts = df['Season'].value_counts()
        for season in self.bn.states['Season']:
            count = season_counts.get(season, 0)
            percentage = (count / len(df)) * 100
            print(f"  {season}: {count} ({percentage:.1f}%)")