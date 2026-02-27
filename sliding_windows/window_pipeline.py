import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matcher import match_windows, evaluate_overlap, plot_windows

def run_synthetic_test():
    np.random.seed(42)
    
    # Generate 5 reference segments
    ref_data = {
        'start_time': [0, 10, 20, 30, 40],
        'end_time': [10, 20, 30, 40, 50],
        'value_a': np.random.uniform(10, 20, 5),
        'value_b': np.random.uniform(100, 200, 5)
    }
    reference_df = pd.DataFrame(ref_data)
    reference_df.index = [f'ref_{i}' for i in range(5)]
    
    # Generate predicted segments: 
    # [0] matches ref_0 (1-to-1)
    # [1] partial match with ref_1
    # [2,3] compete for ref_2 (unique drop test)
    # [4] floating / unassigned
    # [5] full match ref_3
    # [6] low overlap drop filter test for eval threshold
    pred_data = {
        'start_time': [0, 15, 20, 22, 100, 30, 48],
        'end_time':   [9, 25, 28, 30, 110, 40, 52], 
        'value_a': np.random.uniform(10, 20, 7),
        'value_b': np.random.uniform(100, 200, 7)
    }
    predicted_df = pd.DataFrame(pred_data)
    predicted_df.index = [f'pred_{i}' for i in range(7)]
    
    print("=== REFERENCE DATAFRAME ===")
    print(reference_df)
    print("\n=== PREDICTED DATAFRAME ===")
    print(predicted_df)
    print("-" * 50)
    
    print("\n[TEST 1] match_windows(unique_pair=False)")
    pred_copy1 = predicted_df.copy()
    info1 = match_windows(pred_copy1, reference_df, unique_pair=False)
    print("\nPredicted DataFrame after Match:")
    print(pred_copy1[['start_time', 'end_time', 'best_fit', 'overlap']])
    print("\nInfo Dict:", info1)
    
    print("\n" + "=" * 50)
    print("\n[TEST 2] match_windows(unique_pair=True)")
    pred_copy2 = predicted_df.copy()
    info2 = match_windows(pred_copy2, reference_df, unique_pair=True)
    print("\nPredicted DataFrame after Match:")
    print(pred_copy2[['start_time', 'end_time', 'best_fit', 'overlap']])
    print("\nInfo Dict:", info2)
    
    print("\n" + "=" * 50)
    print("\n[TEST 3] evaluate_overlap (threshold=0.4, unique_pair=True setup)")
    eval_results = evaluate_overlap(
        pred_copy2, 
        reference_df, 
        overlap_threshold=0.4, 
        metrics_columns=['value_a', 'value_b']
    )
    print("\nEvaluation Results:")
    for col, res in eval_results.items():
        print(f"[{col}] RMSE={res['RMSE']:.4f}, MAE={res['MAE']:.4f}, MAPE={res['MAPE']:.4f}")

    # Plot the windows
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot reference windows at y=1.0
    plot_windows(ax, reference_df, y_level=1.0, color='lightblue', alpha=0.3)
    
    # Plot predicted windows at y=0.0
    plot_windows(ax, predicted_df, y_level=0.0, color='lightgreen', alpha=0.3)
    
    all_starts = np.concatenate([reference_df['start_time'].values, predicted_df['start_time'].values])
    all_ends = np.concatenate([reference_df['end_time'].values, predicted_df['end_time'].values])
    min_t = min(all_starts) - 10
    max_t = max(all_ends) + 10
    
    ax.set_xlim(min_t, max_t)
    ax.set_ylim(-1.0, 2.0)
    ax.set_yticks([])
    ax.set_xlabel('Time')
    ax.set_title('Sliding Windows Plot')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_synthetic_test()
