import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Signal(object):
    def __init__(self, p_random_gen, length=50):
        self.length = length
        self.half_length = length // 2
        self.full = p_random_gen(length)

    @property
    def left(self):
        return self.full[: self.half_length]

    @property
    def right(self):
        return self.full[self.half_length :]

    def get_side(self, side="full"):
        assert side in ("full", "left", "right"), f"Invalid side: {side}"
        return getattr(self, side)

    def compute_std(self, side=None):
        return np.std(self.get_side(side))

    def show(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        idxs = np.arange(self.length)
        ax.plot(idxs[: self.half_length], self.get_side("left"), label="left")
        ax.plot(idxs[self.half_length :], self.get_side("right"), label="right")
        ax.grid(True)
        ax.legend()
        plt.show()


class SignalGenerator(object):
    def __init__(self, p_random_gen, store=True):
        self.p_random_gen = p_random_gen
        self.signals = list() if store else None

        self.current_signal = None

    def __call__(self, length):
        self.current_signal = Signal(self.p_random_gen, length)
        if isinstance(self.signals, list):
            self.signals.append(self.current_signal)

    def __getitem__(self, item):
        if isinstance(self.signals, list):
            return self.signals[item]
        return None


class RunExperiment(object):
    def __init__(self, signal_generator, lengths, trials_per_length):

        assert len(lengths) >= 1, "At least one length should be choosen"

        self.signal_generator = signal_generator
        self.lengths = lengths
        self.lengths.sort()
        self.trials_per_length = trials_per_length

        self.full_std = dict()
        self.left_std = dict()
        self.right_std = dict()
        self.is_between = dict()
        self.success = dict()
        self.fail = dict()

        for length in tqdm(lengths):
            self.full_std[length] = trials_per_length * [0.0]
            self.left_std[length] = trials_per_length * [0.0]
            self.right_std[length] = trials_per_length * [0.0]
            self.is_between[length] = trials_per_length * [0.0]
            self.success[length] = 0.0

            for trial in tqdm(range(trials_per_length)):
                self.signal_generator(length)
                self.full_std[length][trial] = (
                    self.signal_generator.current_signal.compute_std("full")
                )
                self.left_std[length][trial] = (
                    self.signal_generator.current_signal.compute_std("left")
                )
                self.right_std[length][trial] = (
                    self.signal_generator.current_signal.compute_std("right")
                )

                min_std = min(
                    self.left_std[length][trial], self.right_std[length][trial]
                )
                max_std = max(
                    self.left_std[length][trial], self.right_std[length][trial]
                )

                self.is_between[length][trial] = (
                    min_std <= self.full_std[length][trial] <= max_std
                )

            self.success[length] = (
                np.sum(self.is_between[length]) / self.trials_per_length
            )

    def summary(self):
        print("length: success - fails")
        for length, success in self.success.items():
            fails = 1 - success
            print(f"{length}: {success*100:.2f}% - {fails*100:.2f}%")

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        success_ax = [self.success[length] for length in self.lengths]
        ax.plot(self.lengths, success_ax, marker="o", linestyle=":")
        ax.grid(True)
        ax.set_title("In between success count")
        ax.set_xlabel("Signal length")

    def __call__(self):
        pass
