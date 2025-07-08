import torch
import matplotlib.pyplot as plt
import os
import numpy as np

# Load entropy sets
set1_entropy = torch.load("entropy_set1.pt", map_location="cpu")
set2_entropy = torch.load("entropy_set2.pt", map_location="cpu")

# Output directory
output_dir = "combined_entropy_histograms_cleaned"
os.makedirs(output_dir, exist_ok=True)

# Convert tensors to list safely
def to_list(data):
    return data.tolist() if hasattr(data, 'tolist') else list(data)

# Plot histograms with mean ± 2*std filtering
def plot_cleaned_combined(entropy_dict, set_name, color):
    plt.figure(figsize=(12, 8))
    for exit_name, raw_values in entropy_dict.items():
        values = np.array(to_list(raw_values))
        mean = np.mean(values)
        std = np.std(values)
        filtered = values[(values >= mean - 0.5*std) & (values <= mean + 0.5*std)]


        plt.hist(filtered, bins=50, alpha=0.5, label=exit_name)
    plt.title(f"Filtered Entropy Histogram - {set_name} (Mean ± 2×Std)")
    plt.xlabel("Entropy")
    plt.ylabel("Number of Samples")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{set_name}_filtered_combined.png")
    plt.close()

# Generate both plots
plot_cleaned_combined(set1_entropy, "Set1", "blue")
plot_cleaned_combined(set2_entropy, "Set2", "green")

print(f" Cleaned combined histograms saved to: ./{output_dir}/")
