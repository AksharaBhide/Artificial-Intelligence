"""
Weather Prediction System with UI
Combines Bayesian Network and Fuzzy Logic
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from bayesian_network import WeatherBayesianNetwork
from fuzzy_logic import FuzzyWeatherSystem
from visualizations import WeatherVisualizer
import threading


class WeatherPredictionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🌤️ Intelligent Weather Prediction System")
        self.root.geometry("1200x700")
        self.root.configure(bg='#f0f0f0')

        # Initialize AI systems
        self.bn = WeatherBayesianNetwork()
        self.fuzzy = FuzzyWeatherSystem()
        self.viz = WeatherVisualizer(self.bn)

        # Set style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('TLabel', font=('Arial', 10), background='#f0f0f0')
        self.style.configure('TFrame', background='#f0f0f0')

        self.create_widgets()

    def create_widgets(self):
        """Create all UI widgets"""

        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame, text="🌤️ INTELLIGENT WEATHER PREDICTION SYSTEM",
                                font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)

        # Subtitle with fuzzy logic info
        subtitle = ttk.Label(main_frame, text="Bayesian Network + Fuzzy Logic Integration",
                             font=('Arial', 10, 'italic'))
        subtitle.grid(row=1, column=0, columnspan=3, pady=5)

        # Left Panel - Input Controls
        self.create_input_panel(main_frame)

        # Middle Panel - Results Display
        self.create_results_panel(main_frame)

        # Right Panel - Fuzzy Logic Display
        self.create_fuzzy_panel(main_frame)

        # Bottom Panel - Buttons
        self.create_button_panel(main_frame)

        # Status Bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))

    def create_input_panel(self, parent):
        """Create input controls panel"""
        input_frame = ttk.LabelFrame(parent, text="📥 Input Conditions", padding="10")
        input_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Season selection
        ttk.Label(input_frame, text="Season:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.season_var = tk.StringVar()
        season_combo = ttk.Combobox(input_frame, textvariable=self.season_var,
                                    values=['Spring', 'Summer', 'Fall', 'Winter'],
                                    state='readonly', width=15)
        season_combo.grid(row=0, column=1, pady=5, padx=5)
        season_combo.bind('<<ComboboxSelected>>', self.on_input_change)

        # Temperature (Celsius)
        ttk.Label(input_frame, text="Temperature (°C):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.temp_var = tk.DoubleVar(value=25.0)
        temp_scale = ttk.Scale(input_frame, from_=-10, to=45, variable=self.temp_var,
                               orient=tk.HORIZONTAL, length=150, command=self.on_temp_change)
        temp_scale.grid(row=1, column=1, pady=5, padx=5)
        self.temp_label = ttk.Label(input_frame, text="25.0 °C")
        self.temp_label.grid(row=1, column=2, padx=5)

        # Temperature linguistic terms
        self.temp_terms_frame = ttk.Frame(input_frame)
        self.temp_terms_frame.grid(row=2, column=0, columnspan=3, pady=2)
        self.update_temp_terms(25.0)

        # Humidity (%)
        ttk.Label(input_frame, text="Humidity (%):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.humidity_var = tk.DoubleVar(value=50.0)
        humidity_scale = ttk.Scale(input_frame, from_=0, to=100, variable=self.humidity_var,
                                   orient=tk.HORIZONTAL, length=150, command=self.on_humidity_change)
        humidity_scale.grid(row=3, column=1, pady=5, padx=5)
        self.humidity_label = ttk.Label(input_frame, text="50.0 %")
        self.humidity_label.grid(row=3, column=2, padx=5)

        # Humidity linguistic terms
        self.humidity_terms_frame = ttk.Frame(input_frame)
        self.humidity_terms_frame.grid(row=4, column=0, columnspan=3, pady=2)
        self.update_humidity_terms(50.0)

        # Pressure (hPa)
        ttk.Label(input_frame, text="Pressure (hPa):").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.pressure_var = tk.DoubleVar(value=1013.0)
        pressure_scale = ttk.Scale(input_frame, from_=950, to=1050, variable=self.pressure_var,
                                   orient=tk.HORIZONTAL, length=150, command=self.on_pressure_change)
        pressure_scale.grid(row=5, column=1, pady=5, padx=5)
        self.pressure_label = ttk.Label(input_frame, text="1013.0 hPa")
        self.pressure_label.grid(row=5, column=2, padx=5)

        # Pressure linguistic terms
        self.pressure_terms_frame = ttk.Frame(input_frame)
        self.pressure_terms_frame.grid(row=6, column=0, columnspan=3, pady=2)
        self.update_pressure_terms(1013.0)

    def create_results_panel(self, parent):
        """Create results display panel"""
        results_frame = ttk.LabelFrame(parent, text="📊 Prediction Results", padding="10")
        results_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Create figure for probability chart
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=results_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, pady=10)

        # Results text
        self.result_text = tk.Text(results_frame, height=8, width=40, font=('Courier', 9))
        self.result_text.grid(row=1, column=0, pady=10)

        # Scrollbar for text
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.result_text.configure(yscrollcommand=scrollbar.set)

    def create_fuzzy_panel(self, parent):
        """Create fuzzy logic display panel"""
        fuzzy_frame = ttk.LabelFrame(parent, text="🌀 Fuzzy Logic Analysis", padding="10")
        fuzzy_frame.grid(row=2, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Fuzzy description
        self.fuzzy_text = tk.Text(fuzzy_frame, height=15, width=35, font=('Courier', 9))
        self.fuzzy_text.grid(row=0, column=0, pady=5)

        # Fuzzy rules info
        rules_text = """
Fuzzy Rules Applied:
• Very Hot + Very Dry → Sunny
• Hot + Dry → Sunny
• Cold + Humid → Rainy
• Very Humid + Low Pressure → Stormy
• Mild + Comfortable → Cloudy
• Very Hot + Humid + Low Pressure → Stormy
        """
        ttk.Label(fuzzy_frame, text=rules_text, justify=tk.LEFT,
                  font=('Arial', 8)).grid(row=1, column=0, pady=10)

    def create_button_panel(self, parent):
        """Create button panel"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=3, column=0, columnspan=3, pady=20)

        # Predict button (main action)
        self.predict_btn = ttk.Button(button_frame, text="🔮 PREDICT WEATHER",
                                      command=self.predict_weather,
                                      width=20)
        self.predict_btn.grid(row=0, column=0, padx=10)

        # Visualizations button
        self.viz_btn = ttk.Button(button_frame, text="📊 Show Visualizations",
                                  command=self.show_visualizations,
                                  width=20)
        self.viz_btn.grid(row=0, column=1, padx=10)

        # Clear button
        self.clear_btn = ttk.Button(button_frame, text="🔄 Clear All",
                                    command=self.clear_all,
                                    width=20)
        self.clear_btn.grid(row=0, column=2, padx=10)

        # Exit button
        self.exit_btn = ttk.Button(button_frame, text="❌ Exit",
                                   command=self.root.quit,
                                   width=20)
        self.exit_btn.grid(row=0, column=3, padx=10)

    def update_temp_terms(self, temp):
        """Update temperature linguistic terms display"""
        for widget in self.temp_terms_frame.winfo_children():
            widget.destroy()

        _, _, memberships = self.fuzzy.get_temperature_linguistic(temp)
        sorted_terms = sorted(memberships.items(), key=lambda x: x[1], reverse=True)[:3]

        for i, (term, membership) in enumerate(sorted_terms):
            if membership > 0:
                ttk.Label(self.temp_terms_frame,
                          text=f"{term}: {membership:.2f}",
                          font=('Arial', 8)).grid(row=0, column=i, padx=5)

    def update_humidity_terms(self, humidity):
        """Update humidity linguistic terms display"""
        for widget in self.humidity_terms_frame.winfo_children():
            widget.destroy()

        _, _, memberships = self.fuzzy.get_humidity_linguistic(humidity)
        sorted_terms = sorted(memberships.items(), key=lambda x: x[1], reverse=True)[:3]

        for i, (term, membership) in enumerate(sorted_terms):
            if membership > 0:
                ttk.Label(self.humidity_terms_frame,
                          text=f"{term}: {membership:.2f}",
                          font=('Arial', 8)).grid(row=0, column=i, padx=5)

    def update_pressure_terms(self, pressure):
        """Update pressure linguistic terms display"""
        for widget in self.pressure_terms_frame.winfo_children():
            widget.destroy()

        _, _, memberships = self.fuzzy.get_pressure_linguistic(pressure)
        sorted_terms = sorted(memberships.items(), key=lambda x: x[1], reverse=True)[:3]

        for i, (term, membership) in enumerate(sorted_terms):
            if membership > 0:
                ttk.Label(self.pressure_terms_frame,
                          text=f"{term}: {membership:.2f}",
                          font=('Arial', 8)).grid(row=0, column=i, padx=5)

    def on_temp_change(self, value):
        """Handle temperature slider change"""
        temp = float(value)
        self.temp_label.config(text=f"{temp:.1f} °C")
        self.update_temp_terms(temp)

    def on_humidity_change(self, value):
        """Handle humidity slider change"""
        humidity = float(value)
        self.humidity_label.config(text=f"{humidity:.1f} %")
        self.update_humidity_terms(humidity)

    def on_pressure_change(self, value):
        """Handle pressure slider change"""
        pressure = float(value)
        self.pressure_label.config(text=f"{pressure:.1f} hPa")
        self.update_pressure_terms(pressure)

    def on_input_change(self, event=None):
        """Handle any input change (optional auto-predict)"""
        pass  # Can be enabled if you want auto-predict

    def map_celsius_to_bn_temp(self, celsius):
        """Map Celsius temperature to Bayesian Network temperature category"""
        if celsius < 10:
            return 'Low'
        elif celsius < 25:
            return 'Medium'
        else:
            return 'High'

    def map_humidity_to_bn(self, humidity):
        """Map humidity percentage to Bayesian Network humidity category"""
        if humidity < 30:
            return 'Low'
        elif humidity < 60:
            return 'Medium'
        else:
            return 'High'

    def map_pressure_to_bn(self, pressure):
        """Map pressure in hPa to Bayesian Network pressure category"""
        if pressure < 1000:
            return 'Low'
        elif pressure < 1020:
            return 'Normal'
        else:
            return 'High'

    def predict_weather(self):
        """Main prediction function"""
        self.status_bar.config(text="Predicting...")

        try:
            # Get input values
            season = self.season_var.get()
            temp_c = self.temp_var.get()
            humidity = self.humidity_var.get()
            pressure = self.pressure_var.get()

            if not season:
                messagebox.showwarning("Input Required", "Please select a season!")
                return

            # Map to Bayesian Network categories
            bn_temp = self.map_celsius_to_bn_temp(temp_c)
            bn_humidity = self.map_humidity_to_bn(humidity)
            bn_pressure = self.map_pressure_to_bn(pressure)

            # Get Bayesian Network prediction
            evidence = {
                'Season': season,
                'Temperature': bn_temp,
                'Humidity': bn_humidity,
                'Pressure': bn_pressure
            }

            bn_probs = self.bn.predict_weather(evidence)

            # Get Fuzzy Logic analysis
            fuzzy_analysis = self.fuzzy.get_fuzzy_description(temp_c, humidity, pressure)

            # Display results
            self.display_results(bn_probs, fuzzy_analysis, evidence)

            self.status_bar.config(text="Prediction complete!")

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status_bar.config(text="Error occurred")

    def display_results(self, bn_probs, fuzzy_analysis, evidence):
        """Display prediction results in UI"""

        # Clear previous results
        self.result_text.delete(1.0, tk.END)
        self.fuzzy_text.delete(1.0, tk.END)
        self.ax.clear()

        # Update probability chart
        weathers = list(bn_probs.keys())
        probs = list(bn_probs.values())
        colors = ['#FFD700', '#A9A9A9', '#4682B4', '#483D8B']

        bars = self.ax.bar(weathers, probs, color=colors)
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel('Probability')
        self.ax.set_title('Weather Probability Distribution')

        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{prob:.1%}', ha='center', va='bottom', fontsize=8)

        self.canvas.draw()

        # Update results text
        self.result_text.insert(tk.END, "📊 BAYESIAN NETWORK RESULTS\n")
        self.result_text.insert(tk.END, "=" * 40 + "\n")
        self.result_text.insert(tk.END, f"Evidence:\n")
        for k, v in evidence.items():
            self.result_text.insert(tk.END, f"  {k}: {v}\n")

        self.result_text.insert(tk.END, "\nProbabilities:\n")
        sorted_probs = sorted(bn_probs.items(), key=lambda x: x[1], reverse=True)
        for weather, prob in sorted_probs:
            bar = "█" * int(prob * 30)
            self.result_text.insert(tk.END, f"  {weather:8}: {prob:5.1%} {bar}\n")

        most_likely = sorted_probs[0][0]
        self.result_text.insert(tk.END, f"\n✅ Most likely: {most_likely}\n")

        # Update fuzzy text
        self.fuzzy_text.insert(tk.END, "🌀 FUZZY LOGIC ANALYSIS\n")
        self.fuzzy_text.insert(tk.END, "=" * 40 + "\n\n")

        self.fuzzy_text.insert(tk.END, "Temperature:\n")
        self.fuzzy_text.insert(tk.END, f"  {fuzzy_analysis['temperature']['term']} ")
        self.fuzzy_text.insert(tk.END, f"({fuzzy_analysis['temperature']['membership']:.2f})\n")

        self.fuzzy_text.insert(tk.END, "\nHumidity:\n")
        self.fuzzy_text.insert(tk.END, f"  {fuzzy_analysis['humidity']['term']} ")
        self.fuzzy_text.insert(tk.END, f"({fuzzy_analysis['humidity']['membership']:.2f})\n")

        self.fuzzy_text.insert(tk.END, "\nPressure:\n")
        self.fuzzy_text.insert(tk.END, f"  {fuzzy_analysis['pressure']['term']} ")
        self.fuzzy_text.insert(tk.END, f"({fuzzy_analysis['pressure']['membership']:.2f})\n")

        self.fuzzy_text.insert(tk.END, "\nFuzzy Weather Suggestion:\n")
        fuzzy_weather = fuzzy_analysis['fuzzy_weather']
        for weather, score in sorted(fuzzy_weather.items(), key=lambda x: x[1], reverse=True):
            if score > 0:
                bar = "█" * int(score * 30)
                self.fuzzy_text.insert(tk.END, f"  {weather:8}: {score:5.1%} {bar}\n")

    def show_visualizations(self):
        """Show additional visualizations in new window"""
        viz_window = tk.Toplevel(self.root)
        viz_window.title("Weather Visualizations")
        viz_window.geometry("900x700")

        # Create notebook for tabs
        notebook = ttk.Notebook(viz_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ========== TAB 1: Seasonal Patterns ==========
        frame1 = ttk.Frame(notebook)
        notebook.add(frame1, text="Seasonal Patterns")

        # Create figure for seasonal patterns
        fig1 = plt.Figure(figsize=(8, 5))
        canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create seasonal plot
        ax1 = fig1.add_subplot(111)
        seasons = self.bn.states['Season']
        weather_types = self.bn.states['Weather']
        colors = ['#FFD700', '#A9A9A9', '#4682B4', '#483D8B']

        x = range(len(seasons))
        width = 0.2

        for i, weather in enumerate(weather_types):
            probs = []
            for season in seasons:
                probs.append(self.bn.predict_weather({'Season': season})[weather])
            bars = ax1.bar([xi + i * width for xi in x], probs, width,
                           label=weather, color=colors[i], alpha=0.8)

        ax1.set_xlabel('Season', fontsize=12)
        ax1.set_ylabel('Probability', fontsize=12)
        ax1.set_title('Weather Patterns by Season', fontsize=14, fontweight='bold')
        ax1.set_xticks([xi + width * 1.5 for xi in x])
        ax1.set_xticklabels(seasons)
        ax1.legend(loc='upper right')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        canvas1.draw()

        # ========== TAB 2: Temperature-Humidity Heatmap ==========
        frame2 = ttk.Frame(notebook)
        notebook.add(frame2, text="Temperature-Humidity Heatmap")

        # Create figure for heatmap
        fig2 = plt.Figure(figsize=(8, 6))
        canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create heatmap for each weather type
        temps = self.bn.states['Temperature']
        hums = self.bn.states['Humidity']
        weathers = self.bn.states['Weather']

        # Create 2x2 subplot for 4 weather types
        for idx, weather in enumerate(weathers):
            ax2 = fig2.add_subplot(2, 2, idx + 1)

            # Create matrix for this weather type
            matrix = np.zeros((len(temps), len(hums)))

            for i, temp in enumerate(temps):
                for j, hum in enumerate(hums):
                    probs = self.bn.predict_weather({
                        'Temperature': temp,
                        'Humidity': hum
                    })
                    matrix[i, j] = probs[weather]

            # Create heatmap
            im = ax2.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
            ax2.set_xticks(range(len(hums)))
            ax2.set_yticks(range(len(temps)))
            ax2.set_xticklabels(hums)
            ax2.set_yticklabels(temps)
            ax2.set_title(f'{weather}', fontweight='bold')
            ax2.set_xlabel('Humidity')
            ax2.set_ylabel('Temperature')

            # Add value labels
            for i in range(len(temps)):
                for j in range(len(hums)):
                    text = ax2.text(j, i, f'{matrix[i, j]:.2f}',
                                   ha="center", va="center",
                                   color="black" if matrix[i, j] < 0.7 else "white")

        fig2.suptitle('Weather Probability Heatmaps (Temperature vs Humidity)',
                      fontsize=14, fontweight='bold')
        fig2.tight_layout()
        canvas2.draw()

        # ========== TAB 3: Pressure Effects ==========
        frame3 = ttk.Frame(notebook)
        notebook.add(frame3, text="Pressure Effects")

        fig3 = plt.Figure(figsize=(8, 5))
        canvas3 = FigureCanvasTkAgg(fig3, master=frame3)
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ax3 = fig3.add_subplot(111)
        pressures = self.bn.states['Pressure']

        for weather in weather_types:
            probs = []
            for pressure in pressures:
                probs.append(self.bn.predict_weather({'Pressure': pressure})[weather])
            ax3.plot(pressures, probs, marker='o', linewidth=2, label=weather)

        ax3.set_xlabel('Pressure', fontsize=12)
        ax3.set_ylabel('Probability', fontsize=12)
        ax3.set_title('Effect of Pressure on Weather Probability', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        canvas3.draw()

        # ========== TAB 4: Current Input Analysis ==========
        frame4 = ttk.Frame(notebook)
        notebook.add(frame4, text="Current Input Analysis")

        fig4 = plt.Figure(figsize=(8, 5))
        canvas4 = FigureCanvasTkAgg(fig4, master=frame4)
        canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Get current input values
        season = self.season_var.get() or "Summer"  # Default if none
        temp_c = self.temp_var.get()
        humidity = self.humidity_var.get()
        pressure = self.pressure_var.get()

        # Map to BN categories
        bn_temp = self.map_celsius_to_bn_temp(temp_c)
        bn_humidity = self.map_humidity_to_bn(humidity)
        bn_pressure = self.map_pressure_to_bn(pressure)

        # Get prediction
        evidence = {
            'Season': season,
            'Temperature': bn_temp,
            'Humidity': bn_humidity,
            'Pressure': bn_pressure
        }
        probs = self.bn.predict_weather(evidence)

        # Create pie chart
        ax4 = fig4.add_subplot(121)
        weathers_list = list(probs.keys())
        probs_list = list(probs.values())
        colors_list = ['#FFD700', '#A9A9A9', '#4682B4', '#483D8B']

        # Filter out very small probabilities for cleaner pie
        wedges, texts, autotexts = ax4.pie(probs_list, labels=weathers_list,
                                            colors=colors_list, autopct='%1.1f%%',
                                            startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax4.set_title('Current Prediction', fontweight='bold')

        # Add bar chart for comparison
        ax5 = fig4.add_subplot(122)
        bars = ax5.bar(weathers_list, probs_list, color=colors_list)
        ax5.set_ylim(0, 1)
        ax5.set_ylabel('Probability')
        ax5.set_title('Probability Distribution')
        ax5.grid(True, alpha=0.3)

        # Add value labels
        for bar, prob in zip(bars, probs_list):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{prob:.1%}', ha='center', va='bottom', fontsize=9)

        fig4.suptitle(f'Analysis for: {season}, {temp_c:.0f}°C, {humidity:.0f}% RH, {pressure:.0f} hPa',
                      fontsize=12)
        fig4.tight_layout()
        canvas4.draw()

    def clear_all(self):
        """Clear all inputs and results"""
        self.season_var.set('')
        self.temp_var.set(25.0)
        self.humidity_var.set(50.0)
        self.pressure_var.set(1013.0)

        self.temp_label.config(text="25.0 °C")
        self.humidity_label.config(text="50.0 %")
        self.pressure_label.config(text="1013.0 hPa")

        self.result_text.delete(1.0, tk.END)
        self.fuzzy_text.delete(1.0, tk.END)
        self.ax.clear()
        self.canvas.draw()

        self.update_temp_terms(25.0)
        self.update_humidity_terms(50.0)
        self.update_pressure_terms(1013.0)

        self.status_bar.config(text="Cleared")


def main():
    root = tk.Tk()
    app = WeatherPredictionUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()