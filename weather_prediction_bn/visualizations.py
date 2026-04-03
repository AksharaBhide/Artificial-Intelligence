import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from bayesian_network import WeatherBayesianNetwork


class WeatherVisualizer:
    """Create visualizations for weather predictions"""

    def __init__(self, bn_model):
        self.bn = bn_model
        self.colors = {
            'Sunny': '#FFD700',
            'Cloudy': '#A9A9A9',
            'Rainy': '#4682B4',
            'Stormy': '#483D8B'
        }

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def plot_weather_probabilities(self, title="Weather Probability Distribution"):
        """Plot marginal probabilities of weather types"""
        probs = self.bn.predict_weather()

        plt.figure(figsize=(10, 6))
        bars = plt.bar(probs.keys(), probs.values(), color=[self.colors[w] for w in probs.keys()])

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Weather Type', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=11)

        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        return plt

    def plot_conditional_weather(self, condition_var, condition_value):
        """
        Plot weather probabilities given a condition
        Example: plot_conditional_weather('Season', 'Summer')
        """
        evidence = {condition_var: condition_value}
        probs = self.bn.predict_weather(evidence)

        # Get unconditional probabilities for comparison
        unconditional = self.bn.predict_weather()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Conditional probabilities
        bars1 = ax1.bar(probs.keys(), probs.values(),
                        color=[self.colors[w] for w in probs.keys()])
        ax1.set_title(f'Weather Given {condition_var} = {condition_value}',
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Weather Type', fontsize=11)
        ax1.set_ylabel('Probability', fontsize=11)
        ax1.set_ylim(0, 1)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        # Comparison chart (difference from unconditional)
        differences = {k: probs[k] - unconditional[k] for k in probs.keys()}
        colors_diff = ['green' if diff > 0 else 'red' for diff in differences.values()]

        bars2 = ax2.bar(differences.keys(), differences.values(), color=colors_diff)
        ax2.set_title(f'Change from Unconditional Probabilities',
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('Weather Type', fontsize=11)
        ax2.set_ylabel('Probability Difference', fontsize=11)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:+.3f}', ha='center', va='bottom' if height > 0 else 'top',
                     fontsize=10)

        plt.tight_layout()
        return plt

    def plot_seasonal_weather_patterns(self):
        """Plot weather probabilities for each season"""
        seasons = self.bn.states['Season']

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for i, season in enumerate(seasons):
            probs = self.bn.predict_weather({'Season': season})

            # Create pie chart
            wedges, texts, autotexts = axes[i].pie(
                probs.values(),
                labels=probs.keys(),
                autopct='%1.1f%%',
                colors=[self.colors[w] for w in probs.keys()],
                startangle=90
            )

            # Style the percentage text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            axes[i].set_title(f'{season}', fontsize=14, fontweight='bold')

        plt.suptitle('Weather Patterns by Season', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return plt

    def plot_heatmap(self, var1='Temperature', var2='Humidity'):
        """Create a heatmap of weather probabilities for combinations of two variables"""
        states1 = self.bn.states[var1]
        states2 = self.bn.states[var2]

        # Create a matrix for each weather type
        weather_types = self.bn.states['Weather']

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, weather in enumerate(weather_types):
            matrix = np.zeros((len(states1), len(states2)))

            for i, s1 in enumerate(states1):
                for j, s2 in enumerate(states2):
                    # Get probability of this weather given conditions
                    evidence = {var1: s1, var2: s2}
                    probs = self.bn.predict_weather(evidence)
                    matrix[i, j] = probs[weather]

            # Create heatmap
            sns.heatmap(matrix,
                        annot=True,
                        fmt='.2f',
                        xticklabels=states2,
                        yticklabels=states1,
                        cmap='YlOrRd',
                        ax=axes[idx],
                        cbar_kws={'label': 'Probability'},
                        vmin=0, vmax=1)

            axes[idx].set_title(f'{weather}', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel(var2, fontsize=11)
            axes[idx].set_ylabel(var1, fontsize=11)

        plt.suptitle(f'Weather Probabilities by {var1} and {var2}',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return plt

    def plot_temperature_humidity_interaction(self):
        """Plot how temperature and humidity interact to affect weather"""
        temperatures = self.bn.states['Temperature']
        humidities = self.bn.states['Humidity']

        # Create a 3D bar chart for each weather type
        fig = plt.figure(figsize=(16, 12))

        for idx, weather in enumerate(self.bn.states['Weather']):
            ax = fig.add_subplot(2, 2, idx + 1, projection='3d')

            xpos, ypos = np.meshgrid(range(len(temperatures)), range(len(humidities)))
            xpos = xpos.flatten()
            ypos = ypos.flatten()
            zpos = np.zeros_like(xpos)

            dx = dy = 0.5

            # Get probabilities
            probs = []
            for i, temp in enumerate(temperatures):
                for j, hum in enumerate(humidities):
                    evidence = {'Temperature': temp, 'Humidity': hum}
                    weather_probs = self.bn.predict_weather(evidence)
                    probs.append(weather_probs[weather])

            dz = probs

            # Color based on probability
            colors = plt.cm.RdYlBu(dz)

            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.8)

            ax.set_xlabel('Temperature', fontsize=10)
            ax.set_ylabel('Humidity', fontsize=10)
            ax.set_zlabel('Probability', fontsize=10)
            ax.set_title(f'{weather}', fontsize=12, fontweight='bold')

            ax.set_xticks(range(len(temperatures)))
            ax.set_xticklabels(temperatures)
            ax.set_yticks(range(len(humidities)))
            ax.set_yticklabels(humidities)

        plt.suptitle('3D Visualization of Weather Probabilities',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return plt

    def plot_sensitivity_analysis(self, variable='Temperature'):
        """Plot sensitivity of weather predictions to a variable"""
        states = self.bn.states[variable]
        weather_types = self.bn.states['Weather']

        fig, ax = plt.subplots(figsize=(12, 7))

        x = np.arange(len(states))
        width = 0.2
        multiplier = 0

        for weather in weather_types:
            probs = []
            for state in states:
                evidence = {variable: state}
                weather_probs = self.bn.predict_weather(evidence)
                probs.append(weather_probs[weather])

            offset = width * multiplier
            bars = ax.bar(x + offset, probs, width, label=weather,
                          color=self.colors[weather], alpha=0.8)

            # Add value labels
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{prob:.2f}', ha='center', va='bottom', fontsize=9)

            multiplier += 1

        ax.set_xlabel(variable, fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(f'Sensitivity to {variable}', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(states)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return plt