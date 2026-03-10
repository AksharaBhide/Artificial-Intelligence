"""
Complete Presentation Demo for Weather Prediction Bayesian Network
Run this file to see a full walkthrough of the project
"""

from bayesian_network import WeatherBayesianNetwork
from visualizations import WeatherVisualizer
from data_generator import WeatherDataGenerator
import matplotlib.pyplot as plt
import time
import sys

class WeatherProjectPresentation:
    def __init__(self):
        print("\n" + "="*70)
        print("🌤️  BAYESIAN NETWORK FOR WEATHER PREDICTION")
        print("="*70)

        # Initialize components
        print("\n📋 Loading components...")
        self.bn = WeatherBayesianNetwork()
        self.viz = WeatherVisualizer(self.bn)
        self.generator = WeatherDataGenerator()
        print("✓ Components loaded successfully")

        print("\n📋 This presentation will show:")
        print("   1. Project Introduction")
        print("   2. Network Structure")
        print("   3. Live Probability Calculations")
        print("   4. Interactive Demo")
        print("   5. Visualizations")

    def wait_for_user(self):
        """Better input handling"""
        try:
            input("\n👉 Press ENTER to continue... ")
        except:
            # Fallback if input fails
            print("\n⚠️  Press Ctrl+C to exit if needed")
            time.sleep(2)

    def slide1_introduction(self):
        """Slide 1: Introduction"""
        print("\n" + "="*70)
        print("SLIDE 1: PROJECT INTRODUCTION")
        print("="*70)

        print("\n🎯 Project Objective:")
        print("  • Predict weather conditions using probabilistic reasoning")
        print("  • Model relationships between weather factors")
        print("  • Handle uncertainty in predictions")

        print("\n🧠 AI Technique Used:")
        print("  • Bayesian Inference")
        print("  • Directed Acyclic Graph (DAG)")
        print("  • Conditional Probability Distributions")

        self.wait_for_user()

    def slide2_network_structure(self):
        """Slide 2: Network Structure"""
        print("\n" + "="*70)
        print("SLIDE 2: NETWORK STRUCTURE")
        print("="*70)

        print("\n📊 DAG (Directed Acyclic Graph):")
        print("""
            ┌─────────┐
            │  Season │
            └────┬────┘
         ┌───────┼───────┐
         ▼       ▼       ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │ Temp   │ │ Humidity││Pressure│
    └───┬────┘ └───┬────┘ └───┬────┘
         └──────────┼──────────┘
                    ▼
              ┌──────────┐
              │ Weather  │
              └──────────┘
        """)

        print("\n📋 Variables and States:")
        print("  • Season: Spring, Summer, Fall, Winter")
        print("  • Temperature: Low, Medium, High")
        print("  • Humidity: Low, Medium, High")
        print("  • Pressure: Low, Normal, High")
        print("  • Weather: Sunny, Cloudy, Rainy, Stormy")

        # Show actual visualization
        print("\n📈 Opening network visualization window...")
        print("   (Close the plot window to continue)")
        self.bn.visualize_structure()
        plt.show()

        self.wait_for_user()

    def slide3_probability_calculations(self):
        """Slide 3: Live Probability Calculations"""
        print("\n" + "="*70)
        print("SLIDE 3: LIVE PROBABILITY CALCULATIONS")
        print("="*70)

        # Case 1: No evidence
        print("\n📊 CASE 1: No evidence (Prior Probabilities)")
        probs = self.bn.predict_weather()
        for weather, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 40)
            print(f"  {weather:8}: {prob:5.1%} {bar}")

        self.wait_for_user()

        # Case 2: Summer only
        print("\n☀️ CASE 2: Season = Summer")
        probs = self.bn.predict_weather({'Season': 'Summer'})
        for weather, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 40)
            print(f"  {weather:8}: {prob:5.1%} {bar}")

        self.wait_for_user()

        # Case 3: Summer + High Temperature
        print("\n🌡️ CASE 3: Summer + High Temperature")
        probs = self.bn.predict_weather({'Season': 'Summer', 'Temperature': 'High'})
        for weather, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 40)
            print(f"  {weather:8}: {prob:5.1%} {bar}")

        self.wait_for_user()

        # Case 4: Summer + High Temp + Low Humidity
        print("\n💧 CASE 4: Summer + High Temperature + Low Humidity")
        probs = self.bn.predict_weather({
            'Season': 'Summer',
            'Temperature': 'High',
            'Humidity': 'Low'
        })
        for weather, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 40)
            print(f"  {weather:8}: {prob:5.1%} {bar}")

        self.wait_for_user()

        # Case 5: All factors
        print("\n🌈 CASE 5: Summer + High Temp + Low Humidity + High Pressure")
        probs = self.bn.predict_weather({
            'Season': 'Summer',
            'Temperature': 'High',
            'Humidity': 'Low',
            'Pressure': 'High'
        })
        for weather, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 40)
            print(f"  {weather:8}: {prob:5.1%} {bar}")

        print("\n📈 Notice how Sunny probability increases with favorable conditions!")
        self.wait_for_user()

    def slide4_interactive_demo(self):
        """Slide 4: Interactive Demo - FIXED VERSION"""
        print("\n" + "="*70)
        print("SLIDE 4: INTERACTIVE DEMO")
        print("="*70)

        print("\n🎮 Now you can test your own scenarios!")
        print("   • Type a value and press Enter to include it")
        print("   • Just press Enter to skip that factor")
        print("   • Type 'quit' to exit demo\n")

        demo_count = 0
        while True:
            demo_count += 1
            print("-"*40)
            print(f"Test #{demo_count} (or type 'quit' to exit)")

            # Get user input with clear prompts
            evidence = {}
            valid_inputs = []
            invalid_inputs = []

            # Season input
            season = input("Season (Spring/Summer/Fall/Winter) [Enter to skip]: ").strip()
            if season.lower() == 'quit':
                break
            if season:
                if season in self.bn.states['Season']:
                    evidence['Season'] = season
                    valid_inputs.append(f"Season: {season}")
                else:
                    invalid_inputs.append(f"Season: {season}")

            # Temperature input
            temp = input("Temperature (Low/Medium/High) [Enter to skip]: ").strip()
            if temp.lower() == 'quit':
                break
            if temp:
                if temp in self.bn.states['Temperature']:
                    evidence['Temperature'] = temp
                    valid_inputs.append(f"Temperature: {temp}")
                else:
                    invalid_inputs.append(f"Temperature: {temp}")

            # Humidity input
            humidity = input("Humidity (Low/Medium/High) [Enter to skip]: ").strip()
            if humidity.lower() == 'quit':
                break
            if humidity:
                if humidity in self.bn.states['Humidity']:
                    evidence['Humidity'] = humidity
                    valid_inputs.append(f"Humidity: {humidity}")
                else:
                    invalid_inputs.append(f"Humidity: {humidity}")

            # Pressure input
            pressure = input("Pressure (Low/Normal/High) [Enter to skip]: ").strip()
            if pressure.lower() == 'quit':
                break
            if pressure:
                if pressure in self.bn.states['Pressure']:
                    evidence['Pressure'] = pressure
                    valid_inputs.append(f"Pressure: {pressure}")
                else:
                    invalid_inputs.append(f"Pressure: {pressure}")

            # Show summary of inputs
            if valid_inputs:
                print(f"\n✅ Accepted: {', '.join(valid_inputs)}")
            if invalid_inputs:
                print(f"❌ Invalid (ignored): {', '.join(invalid_inputs)}")

            # Make prediction
            if evidence:
                print(f"\n🔮 Predicting with: {evidence}")
                probs = self.bn.predict_weather(evidence)
            else:
                print("\n📊 No specific conditions - showing base probabilities:")
                probs = self.bn.predict_weather()

            # Display results
            print("\n📊 Results:")
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            for weather, prob in sorted_probs:
                bar = "█" * int(prob * 50)
                print(f"  {weather:8}: {prob:5.1%} {bar}")

            most_likely = sorted_probs[0][0]
            most_likely_prob = sorted_probs[0][1]
            print(f"\n✅ Most likely: {most_likely} ({most_likely_prob:.1%})")

            # Ask if they want to continue
            print("\n" + "-"*20)
            continue_choice = input("Make another prediction? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("\n👋 Exiting interactive demo...")
                break

    def slide5_visualizations(self):
        """Slide 5: Show visualizations"""
        print("\n" + "="*70)
        print("SLIDE 5: VISUALIZATIONS")
        print("="*70)

        # 1. Weather distribution
        print("\n📊 Showing Weather Distribution (close plot to continue)...")
        self.viz.plot_weather_probabilities("Overall Weather Distribution")
        plt.show()

        # 2. Seasonal patterns
        print("\n🍂 Showing Seasonal Patterns (close plot to continue)...")
        self.viz.plot_seasonal_weather_patterns()
        plt.show()

        # 3. Conditional probabilities
        print("\n☀️ Showing Summer Weather (close plot to continue)...")
        self.viz.plot_conditional_weather('Season', 'Summer')
        plt.show()

        # 4. Heatmap
        print("\n🔥 Showing Temperature-Humidity Heatmap (close plot to continue)...")
        self.viz.plot_heatmap('Temperature', 'Humidity')
        plt.show()

        self.wait_for_user()

    def slide6_conclusion(self):
        """Slide 6: Conclusion"""
        print("\n" + "="*70)
        print("SLIDE 6: CONCLUSION")
        print("="*70)

        print("\n✅ Key Achievements:")
        print("  ✓ Built probabilistic weather prediction model")
        print("  ✓ Implemented Bayesian Network with 5 variables")
        print("  ✓ Created interactive prediction system")
        print("  ✓ Developed multiple visualization types")

        print("\n💡 Real-World Applications:")
        print("  • Agriculture - Planting decisions")
        print("  • Aviation - Flight planning")
        print("  • Event Planning - Outdoor scheduling")
        print("  • Disaster Management - Early warnings")

        print("\n" + "="*70)
        print("🎉 THANK YOU!")
        print("="*70)
        print("\nTo explore more:")
        print("  • python main.py --mode interactive  - Make your own predictions")
        print("  • python main.py --mode visualize    - See all visualizations")
        print("  • python main.py --mode data         - Generate synthetic data")

def main():
    """Main function to run presentation"""
    presentation = WeatherProjectPresentation()

    print("\n" + "-"*70)
    presentation.wait_for_user()

    # Run through slides
    presentation.slide1_introduction()
    presentation.slide2_network_structure()
    presentation.slide3_probability_calculations()
    presentation.slide4_interactive_demo()
    presentation.slide5_visualizations()
    presentation.slide6_conclusion()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Presentation ended. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
