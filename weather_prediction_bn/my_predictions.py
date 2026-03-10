# my_predictions.py
from bayesian_network import WeatherBayesianNetwork
from utils import WeatherUtils

# Create the network
bn = WeatherBayesianNetwork()

# Define different scenarios to test
scenarios = [
    {"name": "Typical Summer Day",
     "evidence": {"Season": "Summer", "Temperature": "High", "Humidity": "Medium"}},

    {"name": "Winter Storm Conditions",
     "evidence": {"Season": "Winter", "Temperature": "Low", "Humidity": "High", "Pressure": "Low"}},

    {"name": "Perfect Beach Weather",
     "evidence": {"Season": "Summer", "Temperature": "High", "Humidity": "Low", "Pressure": "High"}},

    {"name": "Rainy Day",
     "evidence": {"Season": "Fall", "Temperature": "Medium", "Humidity": "High", "Pressure": "Low"}},
]

print("=" * 60)
print("WEATHER PREDICTION SCENARIOS")
print("=" * 60)

for scenario in scenarios:
    print(f"\n📊 {scenario['name']}")
    print("-" * 40)
    print(f"Conditions: {scenario['evidence']}")

    # Get prediction
    probs = bn.predict_weather(scenario['evidence'])

    # Sort by probability
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    # Display results
    for weather, prob in sorted_probs:
        bar = "█" * int(prob * 40)
        print(f"  {weather:8}: {prob:5.1%} {bar}")

    # Get advice
    advice = WeatherUtils.get_weather_advice(probs)
    print(f"\n  💡 Advice: {advice['advice']}")
    print(f"  📈 Confidence: {advice['confidence']}")

    # Calculate uncertainty
    uncertainty = WeatherUtils.get_uncertainty_metrics(probs)
    print(f"  🎯 Certainty Score: {uncertainty['certainty_score']:.2f}")