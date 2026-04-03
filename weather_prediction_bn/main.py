#!/usr/bin/env python3
"""
Bayesian Network for Weather Prediction
Main application file
"""

import argparse
import sys
from bayesian_network import WeatherBayesianNetwork
from visualizations import WeatherVisualizer
from data_generator import WeatherDataGenerator
import matplotlib.pyplot as plt


def print_header():
    """Print application header"""
    print("=" * 60)
    print("   BAYESIAN NETWORK FOR WEATHER PREDICTION")
    print("=" * 60)
    print("\nThis application uses Bayesian inference to predict")
    print("weather conditions based on multiple factors.\n")


def interactive_mode():
    """Run interactive prediction mode"""
    bn = WeatherBayesianNetwork()
    viz = WeatherVisualizer(bn)

    print("\n--- INTERACTIVE WEATHER PREDICTION ---")
    print("\nEnter observed conditions (or press Enter to skip):")

    evidence = {}

    # Get Season
    season = input("Season (Spring/Summer/Fall/Winter) [Skip]: ").strip()
    if season and season in bn.states['Season']:
        evidence['Season'] = season

    # Get Temperature
    temp = input("Temperature (Low/Medium/High) [Skip]: ").strip()
    if temp and temp in bn.states['Temperature']:
        evidence['Temperature'] = temp

    # Get Humidity
    hum = input("Humidity (Low/Medium/High) [Skip]: ").strip()
    if hum and hum in bn.states['Humidity']:
        evidence['Humidity'] = hum

    # Get Pressure
    press = input("Pressure (Low/Normal/High) [Skip]: ").strip()
    if press and press in bn.states['Pressure']:
        evidence['Pressure'] = press

    print("\n" + "-" * 40)
    print("PREDICTION RESULTS:")
    print("-" * 40)

    if evidence:
        print(f"Based on: {evidence}")
    else:
        print("No evidence provided - showing prior probabilities")

    probs = bn.predict_weather(evidence)

    print("\nWeather Probabilities:")
    for weather, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 50)
        print(f"  {weather:8}: {prob:.1%} {bar}")

    # Ask if user wants to see visualizations
    print("\n" + "-" * 40)
    show_viz = input("\nShow visualizations? (y/n): ").strip().lower()
    if show_viz == 'y':
        if evidence:
            for var, val in evidence.items():
                viz.plot_conditional_weather(var, val)
                plt.show()
        else:
            viz.plot_weather_probabilities()
            plt.show()


def visualization_demo():
    """Run visualization demo"""
    print("\n--- GENERATING VISUALIZATIONS ---")

    bn = WeatherBayesianNetwork()
    viz = WeatherVisualizer(bn)

    # 1. Show network structure
    print("\n1. Displaying Bayesian Network Structure...")
    bn.visualize_structure()
    plt.show()

    # 2. Basic probability distribution
    print("\n2. Displaying Basic Weather Probability Distribution...")
    viz.plot_weather_probabilities("Marginal Weather Probabilities")
    plt.show()

    # 3. Seasonal patterns
    print("\n3. Displaying Seasonal Weather Patterns...")
    viz.plot_seasonal_weather_patterns()
    plt.show()

    # 4. Conditional probabilities
    print("\n4. Displaying Conditional Probabilities (Summer)...")
    viz.plot_conditional_weather('Season', 'Summer')
    plt.show()

    # 5. Heatmap analysis
    print("\n5. Displaying Temperature-Humidity Heatmap...")
    viz.plot_heatmap('Temperature', 'Humidity')
    plt.show()

    # 6. Sensitivity analysis
    print("\n6. Displaying Sensitivity Analysis...")
    viz.plot_sensitivity_analysis('Temperature')
    plt.show()


def generate_data_demo(n_samples=100):
    """Generate and analyze synthetic data"""
    print(f"\n--- GENERATING {n_samples} SYNTHETIC WEATHER SAMPLES ---")

    generator = WeatherDataGenerator()
    df = generator.generate_dataset(n_samples)

    print("\nSample Data (first 10 rows):")
    print(df.head(10))

    print("\nData Summary:")
    print(df.describe())

    print("\nWeather Distribution:")
    weather_counts = df['Weather'].value_counts()
    for weather, count in weather_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {weather}: {count} ({percentage:.1f}%)")

    return df


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='Bayesian Network for Weather Prediction')
    parser.add_argument('--mode', '-m', choices=['interactive', 'visualize', 'data', 'all'],
                        default='interactive', help='Operation mode')
    parser.add_argument('--samples', '-s', type=int, default=100,
                        help='Number of samples for data generation')

    args = parser.parse_args()

    print_header()

    if args.mode == 'interactive':
        interactive_mode()

    elif args.mode == 'visualize':
        visualization_demo()

    elif args.mode == 'data':
        generate_data_demo(args.samples)

    elif args.mode == 'all':
        print("\nRunning all demonstrations...")

        # Data generation
        df = generate_data_demo(args.samples)

        # Visualizations
        visualization_demo()

        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)