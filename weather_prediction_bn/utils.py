import json
import numpy as np


class WeatherUtils:
    """Utility functions for weather prediction"""

    @staticmethod
    def probability_to_grade(probability):
        """Convert probability to qualitative grade"""
        if probability < 0.2:
            return "Very Unlikely"
        elif probability < 0.4:
            return "Unlikely"
        elif probability < 0.6:
            return "Possible"
        elif probability < 0.8:
            return "Likely"
        else:
            return "Very Likely"

    @staticmethod
    def get_weather_advice(weather_probs):
        """Get advice based on weather predictions"""
        most_likely = max(weather_probs, key=weather_probs.get)
        prob = weather_probs[most_likely]

        advice = {
            'Sunny': "Don't forget sunscreen and stay hydrated!",
            'Cloudy': "Good day for outdoor activities with mild sun.",
            'Rainy': "Better carry an umbrella and wear waterproof shoes.",
            'Stormy': "Stay indoors if possible and secure loose items."
        }

        confidence = WeatherUtils.probability_to_grade(prob)

        return {
            'most_likely_weather': most_likely,
            'confidence': confidence,
            'probability': prob,
            'advice': advice[most_likely]
        }

    @staticmethod
    def save_model(bn_model, filename='weather_bn_model.json'):
        """Save model parameters to JSON"""
        model_data = {
            'structure': list(bn_model.model.edges()),
            'states': bn_model.states
        }

        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)

        print(f"Model saved to {filename}")

    @staticmethod
    def calculate_entropy(probabilities):
        """Calculate entropy of probability distribution"""
        probs = np.array(list(probabilities.values()))
        probs = probs[probs > 0]  # Remove zeros
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    @staticmethod
    def get_uncertainty_metrics(weather_probs):
        """Calculate uncertainty metrics for prediction"""
        entropy = WeatherUtils.calculate_entropy(weather_probs)
        max_prob = max(weather_probs.values())

        # Certainty score (0-1, higher is more certain)
        certainty = 1 - (entropy / np.log2(len(weather_probs)))

        return {
            'entropy': entropy,
            'max_probability': max_prob,
            'certainty_score': certainty
        }