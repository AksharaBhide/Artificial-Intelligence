# batch_predict.py
import csv
from bayesian_network import WeatherBayesianNetwork

bn = WeatherBayesianNetwork()

# Sample data for different cities
cities_data = [
    {"city": "Mumbai", "season": "Summer", "temp": "High", "humidity": "High", "pressure": "Low"},
    {"city": "Dubai", "season": "Summer", "temp": "High", "humidity": "Low", "pressure": "High"},
    {"city": "London", "season": "Fall", "temp": "Low", "humidity": "High", "pressure": "Low"},
    {"city": "Moscow", "season": "Winter", "temp": "Low", "humidity": "Medium", "pressure": "Normal"},
    {"city": "Singapore", "season": "Summer", "temp": "High", "humidity": "High", "pressure": "Normal"},
]

print("=" * 70)
print("WEATHER PREDICTIONS FOR DIFFERENT CITIES")
print("=" * 70)

results = []
for city in cities_data:
    evidence = {
        'Season': city['season'],
        'Temperature': city['temp'],
        'Humidity': city['humidity'],
        'Pressure': city['pressure']
    }

    probs = bn.predict_weather(evidence)
    most_likely = max(probs, key=probs.get)

    result = {
        'City': city['city'],
        'Most Likely Weather': most_likely,
        'Probability': f"{probs[most_likely]:.1%}",
        'Sunny': f"{probs['Sunny']:.1%}",
        'Cloudy': f"{probs['Cloudy']:.1%}",
        'Rainy': f"{probs['Rainy']:.1%}",
        'Stormy': f"{probs['Stormy']:.1%}"
    }
    results.append(result)

    print(f"\n📍 {city['city']}")
    print(
        f"   Conditions: {city['season']}, {city['temp']} temp, {city['humidity']} humidity, {city['pressure']} pressure")
    print(f"   → Most likely: {most_likely} ({probs[most_likely]:.1%})")

# Save to CSV
import pandas as pd

df = pd.DataFrame(results)
df.to_csv('city_predictions.csv', index=False)
print("\n✅ Results saved to 'city_predictions.csv'")