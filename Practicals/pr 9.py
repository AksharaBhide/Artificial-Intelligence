def bayes_rain(prior, sensitivity, false_alarm):
 p_not_rain = 1 - prior
 p_forecast = sensitivity * prior + false_alarm * p_not_rain
 posterior = (sensitivity * prior) / p_forecast
 return p_forecast, posterior
prior_rain = 0.2
sensitivity = 0.9
false_alarm_rate = 0.1
p_forecast, posterior_rain = bayes_rain(prior_rain, sensitivity, false_alarm_rate)
print(f"Prior probability of rain P(R) = {prior_rain}")
print(f"Sensitivity (true positive rate) P(F|R) = {sensitivity}")
print(f"False alarm rate P(F|not R) = {false_alarm_rate}\n")
print("Step 1: Calculate total probability of a rain forecast P(F)")
print(f"P(not R) = 1 - P(R) = {1 - prior_rain}")
print(f"P(F) = P(F|R) * P(R) + P(F|not R) * P(not R)")
print(f" = {sensitivity} * {prior_rain} + {false_alarm_rate} * {1 - prior_rain}")
print(f" = {sensitivity * prior_rain} + {false_alarm_rate * (1 - prior_rain)}")
print(f" = {p_forecast}\n")
print("Step 2: Apply Bayes' rule to find P(R|F)")
print(f"P(R|F) = [P(F|R) * P(R)] / P(F)")
print(f" = ({sensitivity} * {prior_rain}) / {p_forecast}")
print(f" = {sensitivity * prior_rain} / {p_forecast}")
print(f" = {posterior_rain:.4f}")
print(f" ≈ {posterior_rain:.2%}\n")
print(f"Interpretation: Given a rain forecast, the probability of actual rain is about {posterior_rain:.1%}.")