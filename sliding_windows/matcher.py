import numpy as np
import pandas as pd
import matplotlib.patches as patches

def match_windows(predicted, reference, unique_pair=False):
    """
    Matches sliding windows between predicted and reference dataframes based on maximum overlap score.
    
    Parameters:
    - predicted: DataFrame with 'start_time' and 'end_time' columns.
    - reference: DataFrame with 'start_time' and 'end_time' columns.
    - unique_pair: boolean. If True, forces 1-to-1 correspondence.
    
    Returns:
    - info: dictionary with mismatch/paired stats.
      Also modifies 'predicted' DataFrame IN PLACE (adds 'best_fit' and 'overlap').
    """
    if len(predicted) == 0 or len(reference) == 0:
        predicted['best_fit'] = None
        predicted['overlap'] = 0.0
        return {
            'unpaired_reference': reference.index.tolist(),
            'unpaired_predicted': predicted.index.tolist(),
            'dropped_predicted_due_to_non_unique': []
        }
        
    p_starts = predicted['start_time'].values[:, np.newaxis]
    p_ends = predicted['end_time'].values[:, np.newaxis]
    
    r_starts = reference['start_time'].values[np.newaxis, :]
    r_ends = reference['end_time'].values[np.newaxis, :]
    
    overlaps = np.maximum(0, np.minimum(p_ends, r_ends) - np.maximum(p_starts, r_starts))
    unions = np.maximum(p_ends, r_ends) - np.minimum(p_starts, r_starts)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = overlaps / unions
        iou = np.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)
    
    best_fit_pos = np.argmax(iou, axis=1)
    best_overlaps = np.max(iou, axis=1)
    
    ref_indices = reference.index.values[best_fit_pos]
    no_overlap_mask = best_overlaps == 0
    
    # Store indices mapped to best fit
    best_fit_mapped = np.where(no_overlap_mask, None, ref_indices)
    
    predicted['best_fit'] = best_fit_mapped
    predicted['overlap'] = np.where(no_overlap_mask, 0.0, best_overlaps)
    
    dropped_predicted = []
    
    if unique_pair:
        # Filter all current non-null best_fit
        valid_fits = predicted['best_fit'].dropna()
        # Find which reference windows appear multiple times
        counts = valid_fits.value_counts()
        non_unique_refs = counts[counts > 1].index
        
        # Select all predicted rows that are part of this many-to-one conflict
        mask_non_unique = predicted['best_fit'].isin(non_unique_refs)
        dropped_predicted = predicted.index[mask_non_unique].tolist()
        
        # Override values for all competing windows to None and 0 
        predicted.loc[mask_non_unique, 'best_fit'] = None
        predicted.loc[mask_non_unique, 'overlap'] = 0.0
        
    # Build returned summary info
    paired_predicted_mask = predicted['best_fit'].notnull()
    unpaired_predicted = predicted.index[~paired_predicted_mask].tolist()
    
    paired_reference_indices = predicted['best_fit'].dropna().unique()
    unpaired_reference = reference.index[~reference.index.isin(paired_reference_indices)].tolist()
    
    return {
        'unpaired_reference': unpaired_reference,
        'unpaired_predicted': unpaired_predicted,
        'dropped_predicted_due_to_non_unique': dropped_predicted
    }

def evaluate_overlap(predicted, reference, overlap_threshold, metrics_columns):
    """
    Evaluates RMSE, MAE, MAPE directly on paired windows exceeding overlap threshold.
    
    Parameters:
    - predicted: DataFrame with 'best_fit' and 'overlap' metrics columns computed.
    - reference: Complete reference DataFrame.
    - overlap_threshold: float, min overlap required to be kept for metric evaluation.
    - metrics_columns: list of column names used to evaluate error.
    
    Returns:
    - results: dictionary recording evaluation accuracy metrics.
    """
    valid_pred = predicted[
        (predicted['overlap'] >= overlap_threshold) & 
        (predicted['best_fit'].notnull())
    ]
    
    results = {}
    
    if len(valid_pred) == 0:
        for col in metrics_columns:
            results[col] = {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
        return results
        
    ref_indices = valid_pred['best_fit'].values
    
    for col in metrics_columns:
        pred_vals = valid_pred[col].values
        ref_vals = reference.loc[ref_indices, col].values
        
        rmse = float(np.sqrt(np.mean((ref_vals - pred_vals)**2)))
        mae = float(np.mean(np.abs(ref_vals - pred_vals)))
        
        # Standardize 0 references to epsilon for robust parsing division
        eps = np.finfo(float).eps
        mape = float(np.mean(np.abs((ref_vals - pred_vals) / np.where(ref_vals == 0, eps, ref_vals))))
        
        results[col] = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
        
    return results

def plot_windows(ax, df, y_level, height=0.6, color='blue', alpha=0.5, offset=0.6, show_labels=True):
    """
    Plots sliding windows as horizontal boxes on a given matplotlib axis.
    Overlapping windows are vertically displaced.
    
    Parameters:
    - ax: matplotlib.axes.Axes to plot on.
    - df: DataFrame containing 'start_time' and 'end_time' columns.
    - y_level: base y-axis center coordinate for the boxes.
    - height: height of the boxes.
    - color: facecolor of the boxes.
    - alpha: transparency level.
    - offset: vertical displacement applied to overlapping overlapping windows.
    - show_labels: boolean flag to render window indices as text.
    """
    # Sort to place systematically
    sorted_df = df.sort_values('start_time')
    
    # Track the end times of each "level" to find non-overlapping insertion spots. Only use 2 levels.
    levels_ends = [float('-inf'), float('-inf')]
    
    for idx, row in sorted_df.iterrows():
        start = row['start_time']
        end = row['end_time']
        width = end - start
        
        # Try finding available level, otherwise force alternate
        if start >= levels_ends[0]:
            level_idx = 0
        elif start >= levels_ends[1]:
            level_idx = 1
        else:
            # Overlaps both? Force alternate based on whatever ended earliest
            level_idx = 0 if levels_ends[0] <= levels_ends[1] else 1
            
        levels_ends[level_idx] = end
            
        current_y = y_level - (level_idx * offset)

        rect = patches.Rectangle(
            (start, current_y - height / 2), 
            width, height,
            linewidth=1, edgecolor='black', facecolor=color, alpha=alpha
        )
        ax.add_patch(rect)
        
        if show_labels:
            ax.text(start + width / 2, current_y, str(idx), 
                    ha='center', va='center', fontsize=8, color='black', 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
