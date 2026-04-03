"""
Fuzzy Logic Module for Weather Prediction
Adds linguistic variables like "Very Hot", "Slightly Humid", etc.
"""

import numpy as np


class FuzzyWeatherSystem:
    """
    Fuzzy logic system to convert crisp values to linguistic terms
    and provide fuzzy membership values
    """

    def __init__(self):
        # Define membership functions for temperature (°C)
        self.temp_ranges = {
            'Very Cold': (-10, 0, 10),
            'Cold': (5, 10, 15),
            'Cool': (12, 18, 22),
            'Mild': (18, 22, 26),
            'Warm': (22, 26, 30),
            'Hot': (26, 32, 36),
            'Very Hot': (32, 38, 45)
        }

        # Define membership functions for humidity (%)
        self.humidity_ranges = {
            'Very Dry': (0, 15, 30),
            'Dry': (20, 35, 45),
            'Comfortable': (35, 50, 65),
            'Humid': (55, 70, 80),
            'Very Humid': (70, 85, 100)
        }

        # Define membership functions for pressure (hPa)
        self.pressure_ranges = {
            'Very Low': (950, 970, 990),
            'Low': (980, 1000, 1010),
            'Normal': (1005, 1013, 1020),
            'High': (1015, 1025, 1035),
            'Very High': (1030, 1040, 1050)
        }

        # Fuzzy rules for weather prediction
        self.fuzzy_rules = [
            # Rule 1: If very hot and dry -> Sunny
            {'temp': 'Very Hot', 'humidity': 'Very Dry', 'pressure': None, 'result': 'Sunny', 'weight': 0.9},

            # Rule 2: If hot and dry -> Sunny
            {'temp': 'Hot', 'humidity': 'Dry', 'pressure': None, 'result': 'Sunny', 'weight': 0.8},

            # Rule 3: If very cold and humid -> Snow/Rain
            {'temp': 'Very Cold', 'humidity': 'Humid', 'pressure': None, 'result': 'Rainy', 'weight': 0.7},

            # Rule 4: If cold and humid -> Rainy
            {'temp': 'Cold', 'humidity': 'Humid', 'pressure': 'Low', 'result': 'Rainy', 'weight': 0.8},

            # Rule 5: If mild and comfortable -> Cloudy
            {'temp': 'Mild', 'humidity': 'Comfortable', 'pressure': 'Normal', 'result': 'Cloudy', 'weight': 0.6},

            # Rule 6: If very humid and low pressure -> Stormy
            {'temp': None, 'humidity': 'Very Humid', 'pressure': 'Very Low', 'result': 'Stormy', 'weight': 0.9},

            # Rule 7: If hot and very humid -> Stormy
            {'temp': 'Hot', 'humidity': 'Very Humid', 'pressure': 'Low', 'result': 'Stormy', 'weight': 0.8},

            # Rule 8: If very hot and humid -> Cloudy with chance of storm
            {'temp': 'Very Hot', 'humidity': 'Humid', 'pressure': 'Low', 'result': 'Stormy', 'weight': 0.6},

            # Rule 9: If cool and comfortable -> Cloudy
            {'temp': 'Cool', 'humidity': 'Comfortable', 'pressure': 'Normal', 'result': 'Cloudy', 'weight': 0.5},

            # Rule 10: If warm and dry -> Sunny
            {'temp': 'Warm', 'humidity': 'Dry', 'pressure': 'High', 'result': 'Sunny', 'weight': 0.7}
        ]

    def triangular_membership(self, x, a, b, c):
        """Calculate triangular membership value"""
        if x <= a or x >= c:
            return 0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x < c:
            return (c - x) / (c - b)
        else:
            return 0

    def get_temperature_linguistic(self, temp_celsius):
        """Convert temperature to linguistic term with membership"""
        memberships = {}
        for term, (a, b, c) in self.temp_ranges.items():
            memberships[term] = self.triangular_membership(temp_celsius, a, b, c)

        # Return term with highest membership
        best_term = max(memberships, key=memberships.get)
        return best_term, memberships[best_term], memberships

    def get_humidity_linguistic(self, humidity_percent):
        """Convert humidity to linguistic term with membership"""
        memberships = {}
        for term, (a, b, c) in self.humidity_ranges.items():
            memberships[term] = self.triangular_membership(humidity_percent, a, b, c)

        best_term = max(memberships, key=memberships.get)
        return best_term, memberships[best_term], memberships

    def get_pressure_linguistic(self, pressure_hpa):
        """Convert pressure to linguistic term with membership"""
        memberships = {}
        for term, (a, b, c) in self.pressure_ranges.items():
            memberships[term] = self.triangular_membership(pressure_hpa, a, b, c)

        best_term = max(memberships, key=memberships.get)
        return best_term, memberships[best_term], memberships

    def apply_fuzzy_rules(self, temp_term, hum_term, press_term):
        """Apply fuzzy rules to get weather suggestion"""
        weather_scores = {'Sunny': 0, 'Cloudy': 0, 'Rainy': 0, 'Stormy': 0}

        for rule in self.fuzzy_rules:
            match = True
            if rule['temp'] and rule['temp'] != temp_term:
                match = False
            if rule['humidity'] and rule['humidity'] != hum_term:
                match = False
            if rule['pressure'] and rule['pressure'] != press_term:
                match = False

            if match:
                weather_scores[rule['result']] += rule['weight']

        # Normalize scores
        total = sum(weather_scores.values())
        if total > 0:
            for weather in weather_scores:
                weather_scores[weather] /= total

        return weather_scores

    def get_fuzzy_description(self, temp_celsius, humidity_percent, pressure_hpa):
        """Get complete fuzzy description of conditions"""
        temp_term, temp_membership, temp_all = self.get_temperature_linguistic(temp_celsius)
        hum_term, hum_membership, hum_all = self.get_humidity_linguistic(humidity_percent)
        press_term, press_membership, press_all = self.get_pressure_linguistic(pressure_hpa)

        fuzzy_weather = self.apply_fuzzy_rules(temp_term, hum_term, press_term)

        return {
            'temperature': {
                'term': temp_term,
                'membership': temp_membership,
                'all_memberships': temp_all
            },
            'humidity': {
                'term': hum_term,
                'membership': hum_membership,
                'all_memberships': hum_all
            },
            'pressure': {
                'term': press_term,
                'membership': press_membership,
                'all_memberships': press_all
            },
            'fuzzy_weather': fuzzy_weather
        }