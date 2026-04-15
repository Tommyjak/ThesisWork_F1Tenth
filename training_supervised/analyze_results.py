import pandas as pd
import matplotlib.pyplot as plt
import os

def load_results(project_dir, run_name):
    """Loads a YOLO results.csv file and cleans the column names."""
    csv_path = os.path.join(project_dir, run_name, "results.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: Could not find {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    # Strip whitespace from headers (YOLO usually adds spaces like '   val/box_loss')
    df.columns = df.columns.str.strip() 
    return df

def get_training_time(df):
    """Calculates total training duration by automatically detecting if time is cumulative or per-epoch."""
    if df is not None and 'time' in df.columns:
        
        # Check if the time column is strictly getting larger every epoch (Cumulative)
        if df['time'].is_monotonic_increasing:
            total_seconds = df['time'].iloc[-1]  # Grab the final total time from the last row
        else:
            total_seconds = df['time'].sum()     # Sum up all the individual epoch times
            
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m"
        else:
            return f"{int(minutes)}m {int(seconds)}s"
            
    return "Unknown Time"

def plot_comparative_analysis():
    project_dir = "training_experiments"
    
    runs = {
        "Complete (100% BG)": "run_complete",
        "Low BG (30% BG)": "run_low_bg",
        "No BG (0% BG)": "run_no_bg"
    }

    dataframes = {}
    run_times = {}
    
    for label, run_folder in runs.items():
        df = load_results(project_dir, run_folder)
        if df is not None:
            dataframes[label] = df
            # Get the time directly from the dataframe
            run_times[label] = get_training_time(df)

    if not dataframes:
        print("No data found to plot!")
        return

    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("F1TENTH YOLO Background Ratio Comparison", fontsize=16, fontweight='bold')

    # Update: Using the exact column names you provided!
    metrics_to_plot = [
        ('metrics/mAP50(B)', 'mAP@50 (Overall Score)', axes[0, 0]),
        ('metrics/precision(B)', 'Precision (False Positive Resistance)', axes[0, 1]),
        ('metrics/recall(B)', 'Recall (Target Tracking Reliability)', axes[1, 0]),
        ('val/box_loss', 'Validation Box Loss (Bounding Box Tightness)', axes[1, 1])
    ]

    colors = ['blue', 'green', 'red']

    for metric_col, title, ax in metrics_to_plot:
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle='--', alpha=0.7)

        for (label, df), color in zip(dataframes.items(), colors):
            if metric_col in df.columns:
                # Add the exact training time into the legend
                legend_label = f"{label} [{run_times[label]}]"
                ax.plot(df['epoch'], df[metric_col], label=legend_label, color=color, linewidth=2)
        
        ax.legend(loc="lower right")

    plt.tight_layout()
    
    output_img = "background_comparison_dashboard.png"
    plt.savefig(output_img, dpi=300)
    print(f"Analysis complete! Graph saved as {output_img}")
    
    # Terminal Summary
    print("\n--- Training Time Summary ---")
    for label, time_str in run_times.items():
        print(f"{label}: {time_str}")
        
    plt.show()

if __name__ == "__main__":
    plot_comparative_analysis()