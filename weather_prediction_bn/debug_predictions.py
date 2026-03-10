# debug_predictions.py
from bayesian_network import WeatherBayesianNetwork

print("=" * 50)
print("DEBUGGING WEATHER PREDICTIONS")
print("=" * 50)

# Create network
bn = WeatherBayesianNetwork()
print("✓ Network created")

# Test cases
test_cases = [
    {"name": "No evidence", "evidence": {}},
    {"name": "Summer only", "evidence": {"Season": "Summer"}},
    {"name": "Summer + High Temp", "evidence": {"Season": "Summer", "Temperature": "High"}},
    {"name": "Summer + High Temp + Medium Humidity",
     "evidence": {"Season": "Summer", "Temperature": "High", "Humidity": "Medium"}},
]

for test in test_cases:
    print(f"\n📊 {test['name']}")
    print("-" * 40)
    print(f"Evidence: {test['evidence']}")

    probs = bn.predict_weather(test['evidence'])

    for weather, prob in probs.items():
        bar = "█" * int(prob * 40)
        print(f"  {weather:8}: {prob:5.1%} {bar}")

    # Check if all probabilities are equal
    if len(set(probs.values())) == 1:
        print("  ⚠️  WARNING: All probabilities are equal!")