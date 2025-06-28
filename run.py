from src import SignalGenerator, RunExperiment
import numpy as np
import pickle

np.random.seed(42)
lengths = np.arange(10, 501, 10, dtype=int)
trials_per_length = 100000
mu_values = [0, 3, 5, 10]
sigma_values = [1, 1.5, 10, 50]

success = dict()
for mu in mu_values:
    x = SignalGenerator(lambda length: np.random.randn(length) + mu, False)
    experiment = RunExperiment(x, lengths, trials_per_length)
    experiment.summary()
    success[mu] = experiment.success
    print(f"fineshed {mu}")

with open("mu.bin", "wb") as f:
    pickle.dump(success, f)

success = dict()
for sigma in sigma_values:
    x = SignalGenerator(lambda length: sigma * np.random.randn(length), False)
    experiment = RunExperiment(x, lengths, trials_per_length)
    experiment.summary()
    success[sigma] = experiment.success
    print(f"fineshed {sigma}")

with open("sigma.bin", "wb") as f:
    pickle.dump(success, f)

print()
