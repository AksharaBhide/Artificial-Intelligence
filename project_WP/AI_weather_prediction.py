import matplotlib.pyplot as plt

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# Step 1: Define Bayesian Network Structure (DAG)
model = DiscreteBayesianNetwork([
    ('Humidity', 'Rain'),
    ('Wind', 'Rain'),
    ('Temperature', 'Rain')
])


# Step 2: Define Conditional Probability Tables

cpd_humidity = TabularCPD(
    variable='Humidity',
    variable_card=2,
    values=[[0.6], [0.4]],
    state_names={'Humidity': ['Low', 'High']}
)

cpd_wind = TabularCPD(
    variable='Wind',
    variable_card=2,
    values=[[0.7], [0.3]],
    state_names={'Wind': ['Weak', 'Strong']}
)

cpd_temperature = TabularCPD(
    variable='Temperature',
    variable_card=2,
    values=[[0.5], [0.5]],
    state_names={'Temperature': ['Cold', 'Hot']}
)

cpd_rain = TabularCPD(
    variable='Rain',
    variable_card=2,
    values=[
        [0.9, 0.7, 0.6, 0.3, 0.4, 0.2, 0.1, 0.05],
        [0.1, 0.3, 0.4, 0.7, 0.6, 0.8, 0.9, 0.95]
    ],
    evidence=['Humidity', 'Wind', 'Temperature'],
    evidence_card=[2, 2, 2],
    state_names={
        'Rain': ['No', 'Yes'],
        'Humidity': ['Low', 'High'],
        'Wind': ['Weak', 'Strong'],
        'Temperature': ['Cold', 'Hot']
    }
)


# Step 3: Add CPDs
model.add_cpds(cpd_humidity, cpd_wind, cpd_temperature, cpd_rain)

print("Model Valid:", model.check_model())


# Step 4: Inference Engine
infer = VariableElimination(model)


# Step 5: Take User Input
print("\nEnter Weather Conditions")

humidity = input("Humidity (Low/High): ").capitalize()
wind = input("Wind (Weak/Strong): ").capitalize()
temperature = input("Temperature (Cold/Hot): ").capitalize()


# Step 6: Perform Prediction
result = infer.query(
    variables=['Rain'],
    evidence={
        'Humidity': humidity,
        'Wind': wind,
        'Temperature': temperature
    }
)

print("\nRain Prediction Result")
print(result)


# Step 7: Plot Probability Chart
probabilities = result.values
labels = ['No Rain', 'Rain']

plt.bar(labels, probabilities)
plt.title("Rain Prediction Probability")
plt.xlabel("Outcome")
plt.ylabel("Probability")
plt.show()