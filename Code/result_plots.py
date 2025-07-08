import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the combined results file
file_path = "/research/duwe/Sri/Code/table/results.csv"
df = pd.read_csv(file_path, delimiter="\t")

# Ensure directory for plots
os.makedirs("plots", exist_ok=True)

# Plot 1: Exit Index vs Accuracy
plt.figure(figsize=(10, 6))
plt.plot(df["Exit Index"], df["Accuracy (exited) (%)"], 'o-', label='Accuracy per Exit')
plt.xlabel("Exit Index")
plt.ylabel("Accuracy (Exited) (%)")
plt.title("Exit Index vs Accuracy")
plt.grid(True)
plt.legend()
plt.savefig("plots/exit_index_vs_accuracy.png")
plt.close()

# Plot 2: Exit Index vs Samples Exited (%)
plt.figure(figsize=(10, 6))
plt.plot(df["Exit Index"], df["Samples Exited (%)"], 's-', label='Samples Exited')
plt.xlabel("Exit Index")
plt.ylabel("Samples Exited (%)")
plt.title("Exit Index vs Samples Exited (%)")
plt.grid(True)
plt.legend()
plt.savefig("plots/exit_index_vs_samples_exited.png")
plt.close()

# Plot 3: Exit Index vs Total Model Params
plt.figure(figsize=(10, 6))
plt.plot(df["Exit Index"], df["Total Model Params"], '^-', label='Total Model Parameters')
plt.xlabel("Exit Index")
plt.ylabel("Total Model Parameters")
plt.title("Exit Index vs Total Model Params")
plt.grid(True)
plt.legend()
plt.savefig("plots/exit_index_vs_total_params.png")
plt.close()

# Plot 4: Exit Index vs Overall Accuracy
plt.figure(figsize=(10, 6))
plt.plot(df["Exit Index"], df["Overall Accuracy"], 'd-', label='Overall Early-Exit Accuracy')
plt.xlabel("Exit Index")
plt.ylabel("Overall Accuracy")
plt.title("Exit Index vs Overall Early-Exit Accuracy")
plt.grid(True)
plt.legend()
plt.savefig("plots/exit_index_vs_overall_accuracy.png")
plt.close()

# Plot 5: Samples Exited (%) vs Accuracy (Exited)
plt.figure(figsize=(10, 6))
plt.scatter(df["Samples Exited (%)"], df["Accuracy (exited) (%)"], c=df["Exit Index"], cmap='viridis')
plt.xlabel("Samples Exited (%)")
plt.ylabel("Accuracy (Exited) (%)")
plt.title("Samples Exited vs Accuracy (Exited)")
plt.colorbar(label='Exit Index')
plt.grid(True)
plt.savefig("plots/samples_vs_accuracy_scatter.png")
plt.close()

