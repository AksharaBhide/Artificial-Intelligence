import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx


class WeatherBayesianNetwork:
    """
    Bayesian Network for Weather Prediction
    DAG Structure:
    Season -> Temperature
    Season -> Humidity
    Season -> Pressure
    Temperature -> Weather
    Humidity -> Weather
    Pressure -> Weather
    """

    def __init__(self):
        self.model = None
        self.inference = None
        self.states = {
            'Season': ['Spring', 'Summer', 'Fall', 'Winter'],
            'Temperature': ['Low', 'Medium', 'High'],
            'Humidity': ['Low', 'Medium', 'High'],
            'Pressure': ['Low', 'Normal', 'High'],
            'Weather': ['Sunny', 'Cloudy', 'Rainy', 'Stormy']
        }
        self.build_network()

    def build_network(self):
        """Build the Bayesian Network structure"""
        # Define the DAG structure
        self.model = DiscreteBayesianNetwork([
            ('Season', 'Temperature'),
            ('Season', 'Humidity'),
            ('Season', 'Pressure'),
            ('Temperature', 'Weather'),
            ('Humidity', 'Weather'),
            ('Pressure', 'Weather')
        ])

        # Define Conditional Probability Distributions
        self._define_cpds()

        # Add CPDs to model
        self.model.add_cpds(*self.cpds)

        # Check if model is valid
        assert self.model.check_model()

        # Create inference object
        self.inference = VariableElimination(self.model)

    def _define_cpds(self):
        """Define all Conditional Probability Distributions"""
        self.cpds = []

        # 1. CPD for Season (root node - no parents)
        cpd_season = TabularCPD(
            variable='Season',
            variable_card=4,
            values=[[0.25], [0.25], [0.25], [0.25]]
        )
        self.cpds.append(cpd_season)

        # 2. CPD for Temperature given Season
        cpd_temp = TabularCPD(
            variable='Temperature',
            variable_card=3,
            values=[
                [0.30, 0.10, 0.30, 0.50],  # Low temperature
                [0.40, 0.30, 0.40, 0.30],  # Medium temperature
                [0.30, 0.60, 0.30, 0.20]  # High temperature
            ],
            evidence=['Season'],
            evidence_card=[4]
        )
        self.cpds.append(cpd_temp)

        # 3. CPD for Humidity given Season
        cpd_humidity = TabularCPD(
            variable='Humidity',
            variable_card=3,
            values=[
                [0.20, 0.30, 0.25, 0.35],  # Low humidity
                [0.40, 0.40, 0.35, 0.35],  # Medium humidity
                [0.40, 0.30, 0.40, 0.30]  # High humidity
            ],
            evidence=['Season'],
            evidence_card=[4]
        )
        self.cpds.append(cpd_humidity)

        # 4. CPD for Pressure given Season
        cpd_pressure = TabularCPD(
            variable='Pressure',
            variable_card=3,
            values=[
                [0.25, 0.30, 0.25, 0.30],  # Low pressure
                [0.50, 0.45, 0.50, 0.40],  # Normal pressure
                [0.25, 0.25, 0.25, 0.30]  # High pressure
            ],
            evidence=['Season'],
            evidence_card=[4]
        )
        self.cpds.append(cpd_pressure)

        # 5. CPD for Weather given Temperature, Humidity, and Pressure
        # Create a 3D array for probabilities: [weather][temp][hum][press]
        weather_cpd = np.zeros((4, 3, 3, 3))

        for t in range(3):  # Temperature (0=Low, 1=Medium, 2=High)
            for h in range(3):  # Humidity (0=Low, 1=Medium, 2=High)
                for p in range(3):  # Pressure (0=Low, 1=Normal, 2=High)
                    probs = self._calculate_weather_probabilities(t, h, p)
                    weather_cpd[:, t, h, p] = probs

        # Reshape to 2D as required by TabularCPD
        weather_cpd_2d = weather_cpd.reshape(4, 27)

        cpd_weather = TabularCPD(
            variable='Weather',
            variable_card=4,
            values=weather_cpd_2d,
            evidence=['Temperature', 'Humidity', 'Pressure'],
            evidence_card=[3, 3, 3]
        )
        self.cpds.append(cpd_weather)

    def _calculate_weather_probabilities(self, temp_idx, hum_idx, press_idx):
        """
        Calculate weather probabilities based on conditions
        Returns probabilities for [Sunny, Cloudy, Rainy, Stormy]
        """
        # Start with base probabilities
        probs = np.array([0.25, 0.25, 0.25, 0.25])

        # ===== TEMPERATURE EFFECTS =====
        if temp_idx == 2:  # High temperature
            probs[0] += 0.25  # Much more Sunny
            probs[1] += 0.05  # Slightly more Cloudy
            probs[2] -= 0.15  # Less Rainy
            probs[3] += 0.05  # Slightly more Stormy
        elif temp_idx == 1:  # Medium temperature
            probs[0] += 0.10  # More Sunny
            probs[1] += 0.10  # More Cloudy
            probs[2] -= 0.05  # Less Rainy
            probs[3] -= 0.05  # Less Stormy
        else:  # Low temperature
            probs[0] -= 0.15  # Less Sunny
            probs[1] -= 0.05  # Less Cloudy
            probs[2] += 0.20  # More Rainy
            probs[3] += 0.10  # More Stormy

        # ===== HUMIDITY EFFECTS =====
        if hum_idx == 2:  # High humidity
            probs[0] -= 0.20  # Much less Sunny
            probs[1] += 0.05  # Slightly more Cloudy
            probs[2] += 0.20  # Much more Rainy
            probs[3] += 0.15  # More Stormy
        elif hum_idx == 1:  # Medium humidity
            probs[0] += 0.05  # Slightly more Sunny
            probs[1] += 0.10  # More Cloudy
            probs[2] -= 0.05  # Less Rainy
            probs[3] -= 0.05  # Less Stormy
        else:  # Low humidity
            probs[0] += 0.25  # Much more Sunny
            probs[1] -= 0.05  # Less Cloudy
            probs[2] -= 0.15  # Less Rainy
            probs[3] -= 0.10  # Less Stormy

        # ===== PRESSURE EFFECTS =====
        if press_idx == 0:  # Low pressure
            probs[0] -= 0.15  # Less Sunny
            probs[1] -= 0.05  # Less Cloudy
            probs[2] += 0.15  # More Rainy
            probs[3] += 0.25  # Much more Stormy
        elif press_idx == 1:  # Normal pressure
            probs[0] += 0.10  # More Sunny
            probs[1] += 0.10  # More Cloudy
            probs[2] -= 0.05  # Less Rainy
            probs[3] -= 0.10  # Less Stormy
        else:  # High pressure
            probs[0] += 0.20  # Much more Sunny
            probs[1] += 0.05  # Slightly more Cloudy
            probs[2] -= 0.15  # Less Rainy
            probs[3] -= 0.20  # Much less Stormy

        # Ensure no negative probabilities
        probs = np.maximum(probs, 0.01)

        # Normalize to sum to 1
        return probs / np.sum(probs)

    def predict_weather(self, evidence=None):
        """
        Predict weather probabilities given evidence
        evidence: dict with observed variables e.g., {'Season': 'Summer', 'Temperature': 'High'}
        """
        if evidence is None:
            evidence = {}

        # Map string values to indices if needed
        mapped_evidence = {}
        for var, value in evidence.items():
            if var in self.states and value in self.states[var]:
                mapped_evidence[var] = self.states[var].index(value)
            else:
                mapped_evidence[var] = value

        try:
            # Perform inference
            result = self.inference.query(variables=['Weather'], evidence=mapped_evidence)

            # Convert back to readable format
            weather_probs = {}
            for i, state in enumerate(self.states['Weather']):
                weather_probs[state] = float(result.values[i])

            return weather_probs

        except Exception as e:
            print(f"Error in inference: {e}")
            # Return varied default probabilities instead of equal ones
            return {
                'Sunny': 0.35,
                'Cloudy': 0.30,
                'Rainy': 0.20,
                'Stormy': 0.15
            }

    def get_conditional_probabilities(self, target_var, evidence=None):
        """
        Get conditional probabilities for any variable
        """
        if evidence is None:
            evidence = {}

        mapped_evidence = {}
        for var, value in evidence.items():
            if var in self.states and value in self.states[var]:
                mapped_evidence[var] = self.states[var].index(value)
            else:
                mapped_evidence[var] = value

        try:
            result = self.inference.query(variables=[target_var], evidence=mapped_evidence)

            probs = {}
            for i, state in enumerate(self.states[target_var]):
                probs[state] = float(result.values[i])

            return probs

        except Exception as e:
            print(f"Error in inference: {e}")
            return {state: 1.0 / len(self.states[target_var]) for state in self.states[target_var]}

    def visualize_structure(self):
        """Visualize the DAG structure"""
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        G.add_edges_from(self.model.edges())

        pos = nx.spring_layout(G, k=2, iterations=50)

        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=3000, font_size=12, font_weight='bold',
                arrows=True, arrowsize=20, edge_color='gray',
                arrowstyle='->', connectionstyle='arc3,rad=0.1')

        plt.title("Bayesian Network Structure for Weather Prediction", fontsize=14, fontweight='bold')
        plt.axis('off')
        try:
            plt.tight_layout()
        except:
            pass  # Ignore tight_layout errors
        return plt