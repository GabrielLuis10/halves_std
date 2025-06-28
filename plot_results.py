import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_from_bin(file_name, lengths):    
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    
    fig, ax = plt.subplots()
    for key, success in data.items():
        success_ax = [success[length] for length in lengths]
        ax.plot(lengths, success_ax, marker='o', linestyle=':', label=key)
    
    ax.set_title(file_name)
    ax.set_xlabel('Signal length')
    ax.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    lengths = np.arange(10, 501, 10, dtype=int)

    plot_from_bin('mu.bin', lengths)
    plot_from_bin('sigma.bin', lengths)
