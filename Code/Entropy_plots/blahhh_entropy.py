import torch
import matplotlib.pyplot as plt
import os

# Load .pt entropy data
set1_entropy = torch.load("entropy_set1.pt", map_location="cpu")
set2_entropy = torch.load("entropy_set2.pt", map_location="cpu")

# Create folder
os.makedirs("entropy_histograms", exist_ok=True)

def plot_entropy(entropy_data, set_name, color):
    for exit_name, entropy in entropy_data.items():
        if hasattr(entropy, 'tolist'):
            entropy = entropy.tolist()
        plt.figure(figsize=(8, 5))
        plt.hist(entropy, bins=50, alpha=0.75, color=color)
        plt.title(f"{set_name} - {exit_name}")
        plt.xlabel("Entropy")
        plt.ylabel("Number of Samples")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"entropy_histograms/{set_name}_{exit_name}.png")
        plt.close()

# Generate histograms
plot_entropy(set1_entropy, "Set1", "blue")
plot_entropy(set2_entropy, "Set2", "green")

print(" Plots saved in 'entropy_histograms/' folder.")
