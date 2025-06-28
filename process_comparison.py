from src import SignalGenerator, RunExperiment
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
lengths = np.arange(10, 201, 10, dtype=int)
trials_per_length = 100000

gaussian_process = SignalGenerator(np.random.randn, False)
poisson_process = SignalGenerator(lambda length: np.random.poisson(1, length), False)
uniform_process = SignalGenerator(np.random.random_sample, False)

gaussian_experiment = RunExperiment(gaussian_process, lengths, trials_per_length)
poisson_experiment = RunExperiment(poisson_process, lengths, trials_per_length)
uniform_experiment = RunExperiment(uniform_process, lengths, trials_per_length)

fig, ax = plt.subplots()
gaussian_experiment.plot(ax)
poisson_experiment.plot(ax)
uniform_experiment.plot(ax)

ax.legend(["gaussian", "poisson", 'uniform'])
plt.show()
