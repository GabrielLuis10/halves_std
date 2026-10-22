import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ==========================================
# STAGE 1: Calculate Metrics
# ==========================================
def calculate_metrics(error_array):
    """Computes useful features from an array of absolute errors."""
    return {
        'mean': np.mean(error_array),
        'max': np.max(error_array),
        'min': np.min(error_array),
        'std': np.std(error_array),
        'median': np.median(error_array)
    }

def process_error_dict(error_dict):
    """
    Recursively traverses the nested dictionary.
    Replaces leaf numpy arrays with a dictionary of computed metrics.
    """
    metrics_dict = {}
    for key, value in error_dict.items():
        if isinstance(value, dict):
            # If value is a dictionary, recurse deeper
            metrics_dict[key] = process_error_dict(value)
        elif isinstance(value, (np.ndarray, list)):
            # If value is an array, compute the metrics
            metrics_dict[key] = calculate_metrics(np.array(value))
        else:
            raise TypeError(f"Expected dict or numpy array, got {type(value)}")
            
    return metrics_dict


# ==========================================
# STAGE 2: Convert to Pandas DataFrame
# ==========================================
def _flatten_metrics_dict(d, current_path=None):
    """Helper generator to flatten the nested metrics dictionary into rows."""
    if current_path is None:
        current_path = []
        
    for key, value in d.items():
        if isinstance(value, dict):
            # Check if this is the final leaf dictionary containing our metrics
            is_metric_leaf = all(not isinstance(v, dict) for v in value.values())
            
            if is_metric_leaf:
                # Build the row: First add the nested keys as columns
                row = {f'Key_Level_{i}': k for i, k in enumerate(current_path + [key])}
                # Then add the computed metrics as columns
                row.update(value)
                yield row
            else:
                # Still navigating the nested structure, recurse deeper
                yield from _flatten_metrics_dict(value, current_path + [key])

def dict_to_dataframe(metrics_dict):
    """Converts the nested metrics dictionary into a Pandas DataFrame."""
    rows = list(_flatten_metrics_dict(metrics_dict))
    return pd.DataFrame(rows)

# ==========================================
# STAGE 3: Generate Plots (Matplotlib)
# ==========================================
def plot_metrics(df, main_metric='mean', error_metric='std'):
    """
    Generates a horizontal bar chart of a target metric for each nested path.
    Includes error bars if the standard deviation is provided.
    """
    # 1. Dynamically find all 'Key_Level' columns
    key_cols = [col for col in df.columns if col.startswith('Key_Level_')]
    
    # 2. Combine the nested keys into a single readable label path
    # e.g., "Model_A > Scenario_1 > Trial_1"
    labels = df[key_cols].astype(str).agg(' > '.join, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.5)))
    y_pos = range(len(df))
    
    # 3. Plot the data with or without error bars
    if error_metric and error_metric in df.columns:
        ax.barh(y_pos, df[main_metric], xerr=df[error_metric], 
                capsize=5, color='steelblue', alpha=0.8, edgecolor='black')
    else:
        ax.barh(y_pos, df[main_metric], 
                color='steelblue', alpha=0.8, edgecolor='black')

    # 4. Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # Reads top-to-bottom matching the DataFrame order
    
    ax.set_xlabel(f'Error ({main_metric.capitalize()})')
    ax.set_title(f'Error Analysis: {main_metric.capitalize()} per Configuration')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

    
def plot_ridgeline(df, metric='mean', group_by_col='Key_Level_0', overlap=1.5, file_name=None):
    """
    Generates a Ridgeline plot of a specific metric, grouped by a column.
    Requires multiple data points per group to compute the density.
    """
    # 1. Identify the unique groups based on the chosen column
    if group_by_col not in df.columns:
        raise ValueError(f"Column '{group_by_col}' not found. Available: {list(df.columns)}")
        
    groups = df[group_by_col].unique()
    
    # 2. Define the common X-axis range with some padding
    x_min, x_max = df[metric].min(), df[metric].max()
    x_padding = (x_max - x_min) * 0.5
    if x_padding == 0: 
        x_padding = 0.1 # Fallback if all values are identical
    x_eval = np.linspace(x_min - x_padding, x_max + x_padding, 500)

    fig, ax = plt.subplots(figsize=(10, max(4, len(groups) * 1.2)))
    
    # 3. Plot each group bottom-to-top to ensure proper overlapping (zorder)
    y_ticks = []
    y_tick_labels = []
    
    for i, group in enumerate(reversed(groups)):
        group_data = df[df[group_by_col] == group][metric].values
        
        # KDE requires variance and >1 data points
        if len(group_data) > 1 and np.std(group_data) > 0:
            kde = gaussian_kde(group_data)
            y_density = kde(x_eval)
            
            # Normalize height based on the chosen overlap factor
            y_scaled = (y_density / np.max(y_density)) * overlap
            y_offset = y_scaled + i
            
            # Fill under the curve and plot the outline
            ax.fill_between(x_eval, i, y_offset, alpha=0.7, zorder=i)
            ax.plot(x_eval, y_offset, color='black', lw=1.2, zorder=i)
        else:
            # Fallback for single data points: plot a vertical line instead of density
            for val in group_data:
                ax.vlines(val, ymin=i, ymax=i + overlap * 0.5, color='black', lw=2, zorder=i)
        
        # Track baseline positions for the Y-axis labels
        y_ticks.append(i)
        y_tick_labels.append(group)


    # 4. Clean up axes (remove spines for the classic ridgeline look)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_xlabel(f'{metric.capitalize()} Error')
    ax.set_title(f'Distribution of {metric.capitalize()} by {group_by_col}')
    
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_tick_params(length=0) # Hide Y ticks marks, keep labels
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    
    return fig, ax

# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    # 1. Create a dummy nested dictionary with 3 levels of depth
    np.random.seed(42)
    mock_errors = {
        'Model_A': {
            'Scenario_1': {
                'Trial_1': np.random.rand(100),
                'Trial_2': np.random.rand(100)
            },
            'Scenario_2': {
                'Trial_1': np.random.rand(100)
            }
        },
        'Model_B': {
            'Scenario_1': {
                'Trial_1': np.random.rand(100)
            }
        }
    }

    # Execute Stage 1
    stage_1_output = process_error_dict(mock_errors)
    
    # Execute Stage 2
    final_df = dict_to_dataframe(stage_1_output)
    print(final_df)
    
    # Plot the mean error with standard deviation as error bars
    fig, ax = plot_metrics(final_df, main_metric='mean', error_metric='std')
    
    # Or plot just the maximum error
    # fig, ax = plot_metrics(final_df, main_metric='max', error_metric=None)

    # To test this effectively, ensure your dummy dictionary generates 
    # enough trials under Scenario_1 and Scenario_2 to form a distribution.
    
    # Plot the distribution of the 'mean' metric, grouped by the root key (Model_A vs Model_B)
    fig, ax = plot_ridgeline(final_df, metric='mean', group_by_col='Key_Level_0')
    
    # Or group by the second level (Scenario_1 vs Scenario_2)
    # fig, ax = plot_ridgeline(final_df, metric='max', group_by_col='Key_Level_1')