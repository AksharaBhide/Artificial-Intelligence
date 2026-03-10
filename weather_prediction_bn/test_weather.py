#!/usr/bin/env python3
"""
Test script for Weather Prediction Bayesian Network
Run this first to verify everything is working
"""

import sys
import numpy as np
from bayesian_network import WeatherBayesianNetwork
from data_generator import WeatherDataGenerator


def print_separator():
    print("\n" + "=" * 60)


def test_imports():
    """Test if all required packages are installed"""
    print("Step 1: Testing imports...")
    try:
        import numpy
        import pandas
        import matplotlib
        import seaborn
        import pgmpy
        import networkx
        import scipy
        print("✓ All packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_bayesian_network():
    """Test Bayesian Network creation and predictions"""
    print("\nStep 2: Testing Bayesian Network...")

    try:
        # Create network
        bn = WeatherBayesianNetwork()
        print("✓ Network created successfully")

        # Test prediction without evidence
        print("\n  Testing prediction (no evidence):")
        probs = bn.predict_weather()
        for weather, prob in probs.items():
            print(f"    {weather}: {prob:.3f}")

        # Test prediction with evidence
        print("\n  Testing prediction (Summer, High temperature):")
        probs = bn.predict_weather({'Season': 'Summer', 'Temperature': 'High'})
        for weather, prob in probs.items():
            print(f"    {weather}: {prob:.3f}")

        # Test conditional probabilities
        print("\n  Testing conditional probabilities:")
        probs = bn.get_conditional_probabilities('Temperature', {'Season': 'Winter'})
        print(f"    Temperature in Winter: {probs}")

        return bn

    except Exception as e:
        print(f"✗ Error in Bayesian Network: {e}")
        return None


def test_data_generator(bn):
    """Test data generation"""
    print("\nStep 3: Testing Data Generator...")

    try:
        generator = WeatherDataGenerator()
        generator.bn = bn  # Use the same network

        # Generate small dataset
        df = generator.generate_dataset(50)
        print(f"✓ Generated {len(df)} samples successfully")

        # Show sample
        print("\n  First 5 samples:")
        print(df.head())

        # Show distribution
        print("\n  Weather distribution:")
        weather_dist = df['Weather'].value_counts()
        for weather, count in weather_dist.items():
            percentage = (count / len(df)) * 100
            print(f"    {weather}: {count} ({percentage:.1f}%)")

        return df

    except Exception as e:
        print(f"✗ Error in Data Generator: {e}")
        return None


def test_visualization(bn):
    """Test if visualization functions work"""
    print("\nStep 4: Testing Visualizations...")

    try:
        from visualizations import WeatherVisualizer

        viz = WeatherVisualizer(bn)
        print("✓ Visualizer created successfully")

        # Just test creation, don't show plots
        print("  ✓ Plot functions available:")
        print("    - plot_weather_probabilities()")
        print("    - plot_conditional_weather()")
        print("    - plot_seasonal_weather_patterns()")
        print("    - plot_heatmap()")
        print("    - plot_sensitivity_analysis()")

        return viz

    except Exception as e:
        print(f"✗ Error in Visualizations: {e}")
        return None


def test_main_script():
    """Test if main.py runs without errors"""
    print("\nStep 5: Testing main script import...")

    try:
        import main
        print("✓ main.py imported successfully")
        return True
    except Exception as e:
        print(f"✗ Error importing main: {e}")
        return False


def run_final_check():
    """Run all tests and provide summary"""
    print("\n" + "=" * 60)
    print("WEATHER PREDICTION SYSTEM - COMPLETE TEST")
    print("=" * 60)

    tests_passed = 0
    total_tests = 5

    # Test 1: Imports
    if test_imports():
        tests_passed += 1

    # Test 2: Bayesian Network
    bn = test_bayesian_network()
    if bn:
        tests_passed += 1

    # Test 3: Data Generator
    df = test_data_generator(bn)
    if df is not None:
        tests_passed += 1

    # Test 4: Visualizations
    viz = test_visualization(bn)
    if viz:
        tests_passed += 1

    # Test 5: Main script
    if test_main_script():
        tests_passed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {tests_passed}/{total_tests} tests passed")
    print("=" * 60)

    if tests_passed == total_tests:
        print("\n✅ SUCCESS! All systems are ready!")
        print("\nYou can now run the main application:")
        print("  python main.py --mode interactive  (for interactive predictions)")
        print("  python main.py --mode visualize    (for visualizations)")
        print("  python main.py --mode data --samples 200  (for data generation)")
        print("  python main.py --mode all          (for complete demo)")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

    return tests_passed == total_tests


if __name__ == "__main__":
    try:
        success = run_final_check()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)