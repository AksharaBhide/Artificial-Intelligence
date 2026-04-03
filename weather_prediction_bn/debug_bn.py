# debug_bn.py
from bayesian_network import WeatherBayesianNetwork

print("="*50)
print("DEBUGGING BAYESIAN NETWORK")
print("="*50)

# Create instance
bn = WeatherBayesianNetwork()

# Test Case: Summer, Very Hot (maps to High), Humid (maps to High), Very High Pressure (maps to High)
evidence = {
    'Season': 'Summer',
    'Temperature': 'High',
    'Humidity': 'High',
    'Pressure': 'High'
}

print(f"\nTesting with evidence: {evidence}")
probs = bn.predict_weather(evidence)

print("\nResults:")
for weather, prob in probs.items():
    print(f"  {weather}: {prob:.3f} ({prob:.1%})")

# Check if all probabilities are equal
if len(set(probs.values())) == 1:
    print("\n❌ PROBLEM: All probabilities are equal!")
    print("   The evidence is not affecting the prediction.")
else:
    print("\n✅ GOOD: Probabilities are varied.")