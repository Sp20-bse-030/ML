import tensorflow_probability as tfp
import tensorflow as tf
import tf_keras as tfk


# Alias for TensorFlow Probability distributions
tfd = tfp.distributions
# Define the initial distribution for the states (cold or hot days)
initial_centers = tfd.Categorical(probs=[0.8, 0.2])
# Define the transition probabilities between states (cold or hot days)
transistion_centers = tfd.Categorical(probs=[[0.7, 0.3], [0.2, 0.8]])
# Define the observation distribution for temperatures on cold and hot days
observation_distribution = tfd.Normal(loc=[10, 25], scale=[5, 10])


# Create the Hidden Markov Model (HMM)
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_centers,
    transition_distribution=transistion_centers,
    observation_distribution=observation_distribution,
    num_steps=7
)

# Compute the expected mean temperature for each of the next 7 days
mean = model.mean()
formatted_temps = [f"Day {i + 1}: {temp:.2f}°C" for i, temp in enumerate(mean.numpy())]  # Format with "°C" and day number
print("Next seven days expected temperatures:")
print("\n".join(formatted_temps))