import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal, interpolate

# --- 1. Preprocessing (Run Once) ---

def align_and_resample(y1, fs1, y2, fs2):
    """
    Aligns two signals to the same time axis.
    Returns: t_common, y1_aligned, y2_aligned
    """
    # Duration and Time Vectors
    dur1, dur2 = len(y1) / fs1, len(y2) / fs2
    min_dur = min(dur1, dur2)
    
    # Target Sampling Rate (Min to avoid interpolation artifacts)
    target_fs = min(fs1, fs2)
    num_samples = int(min_dur * target_fs)
    t_common = np.linspace(0, min_dur, num_samples)
    
    # Interpolators
    t1 = np.linspace(0, dur1, len(y1))
    t2 = np.linspace(0, dur2, len(y2))
    
    # specific 'kind' can be cubic, but linear is safer for noisy sensor data
    f1 = interpolate.interp1d(t1, y1, kind='linear', fill_value="extrapolate")
    f2 = interpolate.interp1d(t2, y2, kind='linear', fill_value="extrapolate")
    
    return t_common, f1(t_common), f2(t_common)

# --- 2. Metric Calculations (On Aligned Data) ---

def compute_metrics(y1, y2):
    """
    Computes concordance metrics on already aligned arrays.
    """
    # Pearson (Correlation)
    r, _ = stats.pearsonr(y1, y2)
    
    # RMSE (Error Magnitude)
    rmse = np.sqrt(np.mean((y1 - y2) ** 2))
    
    # Lin's Concordance Correlation Coefficient (CCC)
    mu_x, mu_y = np.mean(y1), np.mean(y2)
    var_x, var_y = np.var(y1), np.var(y2)
    sd_x, sd_y = np.std(y1), np.std(y2)
    numerator = 2 * r * sd_x * sd_y
    denominator = var_x + var_y + (mu_x - mu_y)**2
    ccc = numerator / denominator
    
    return {"Pearson_r": r, "RMSE": rmse, "Lins_CCC": ccc}

# --- 3. Visualization Functions ---

def plot_bland_altman(y1, y2, title="Bland-Altman Plot"):
    """
    Generates a Bland-Altman plot to detect systematic bias.
    """
    means = (y1 + y2) / 2
    diffs = y1 - y2
    mean_bias = np.mean(diffs)
    std_diff = np.std(diffs)
    
    upper_loa = mean_bias + 1.96 * std_diff
    lower_loa = mean_bias - 1.96 * std_diff
    
    plt.figure(figsize=(10, 6))
    plt.scatter(means, diffs, alpha=0.5, c='teal', label='Samples')
    
    # Plot horizontal lines for Bias and LoA
    plt.axhline(mean_bias, color='red', linestyle='-', label=f'Mean Bias ({mean_bias:.2f})')
    plt.axhline(upper_loa, color='gray', linestyle='--', label=f'+1.96 SD ({upper_loa:.2f})')
    plt.axhline(lower_loa, color='gray', linestyle='--', label=f'-1.96 SD ({lower_loa:.2f})')
    
    # Shaded confidence area
    plt.axhspan(lower_loa, upper_loa, color='gray', alpha=0.1)
    
    plt.title(title)
    plt.xlabel('Mean of two measures')
    plt.ylabel('Difference (Device 1 - Device 2)')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()

def plot_signal_comparison(t, y1, y2):
    """
    Standard Time-Series overlay.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(t, y1, label='Device 1', alpha=0.8)
    plt.plot(t, y2, label='Device 2', alpha=0.8, linestyle='--')
    plt.title("Signal Alignment Check")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 4. Main Workflow ---

def analyze_concordance(y1_raw, fs1, y2_raw, fs2):
    print("1. Aligning and Resampling signals...")
    t, y1, y2 = align_and_resample(y1_raw, fs1, y2_raw, fs2)
    
    print("2. Computing Metrics...")
    metrics = compute_metrics(y1, y2)
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")
        
    print("3. Generating Plots...")
    plot_signal_comparison(t, y1, y2)
    plot_bland_altman(y1, y2)

# --- Example Run ---
if __name__ == "__main__":
    # Create dummy data: 
    # Device A is 100Hz, Device B is 60Hz with some noise and offset
    fs_a, fs_b = 100, 60
    t_a = np.linspace(0, 10, fs_a * 10)
    t_b = np.linspace(0, 10, fs_b * 10)
    
    sig_a = np.sin(t_a) + np.sin(2 * np.pi * 0.5 * t_a)
    sig_b = (np.sin(t_b) + np.sin(2 * np.pi * 0.5 * t_b)) * 1.05 + 0.1 + np.random.normal(0, 0.1, len(t_b))
    
    analyze_concordance(sig_a, fs_a, sig_b, fs_b)