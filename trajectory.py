import numpy as np
import matplotlib.pyplot as plt

def tricky_traj(t):
    # A spiral-like trajectory where curvature increases over time
    # This makes it harder for lower sampling rates to "keep up" with the sharp turns
    x = t * np.cos((t/10)**2)
    y = t * np.sin((t/10)**2)
    return x, y

def plot_sampling_resolution_tricky(trajectory_func, sampling_rates, t_window):
    # Generate fine trajectory for background
    t_fine = np.linspace(t_window[0], t_window[1], 2000)
    x_fine, y_fine = trajectory_func(t_fine)
    
    plt.figure(figsize=(10, 8))
    plt.plot(x_fine, y_fine, 'k-', alpha=0.2, label='Actual Trajectory', linewidth=2)
    
    # Plot for each rate
    for rate in sampling_rates:
        dt = 1.0 / rate
        t_sample = np.arange(t_window[0], t_window[1], dt)
        x_sample, y_sample = trajectory_func(t_sample)
        
        # Displaying with markers
        plt.plot(x_sample, y_sample, marker='o', markersize=5, linestyle='--', label=f'{rate} Hz')
        
    plt.title("Sampling Effect on a Curving Trajectory (Tricky for Lower Rates)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    plt.show()

# 30, 80, 100 Hz
# Time window [0, 3] covers a segment where the spiral tightens
plot_sampling_resolution_tricky(tricky_traj, [45, 80, 100], (0, 0.07))