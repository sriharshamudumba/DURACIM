import matplotlib.pyplot as plt
import pandas as pd

# Data manually extracted from the user's screenshot
data = {
    'Residual Block': [
        "exit_blocks_0_4", "exit_blocks_0_5", "exit_blocks_0_6", "exit_blocks_0_7", "exit_blocks_0_8",
        "exit_blocks_0_9", "exit_blocks_0_10", "exit_blocks_0_11", "exit_blocks_0_12", "exit_blocks_0_13", "exit_blocks_0_14",
        "exit_blocks_1_4", "exit_blocks_1_5", "exit_blocks_1_6", "exit_blocks_1_7", "exit_blocks_1_8",
        "exit_blocks_1_9", "exit_blocks_1_10", "exit_blocks_1_11", "exit_blocks_1_12", "exit_blocks_1_13", "exit_blocks_1_14",
        "exit_blocks_2_4", "exit_blocks_2_5", "exit_blocks_2_6", "exit_blocks_2_7", "exit_blocks_2_8",
        "exit_blocks_2_9", "exit_blocks_2_10", "exit_blocks_2_11", "exit_blocks_2_12", "exit_blocks_2_13", "exit_blocks_2_14",
        "exit_blocks_3_4", "exit_blocks_3_5", "exit_blocks_3_6", "exit_blocks_3_7", "exit_blocks_3_8",
        "exit_blocks_3_9", "exit_blocks_3_10", "exit_blocks_3_11", "exit_blocks_3_12", "exit_blocks_3_13", "exit_blocks_3_14"
    ],
    'ForcedExit1': [
        26.93, 25.88, 26.02, 26.01, 26.24, 26.46, 26.72, 26.53, 26.82, 26.26, 25.59,
        30.50, 30.32, 30.09, 30.15, 30.27, 29.99, 29.79, 29.00, 30.00, 29.74, 29.82,
        31.20, 31.98, 30.82, 30.04, 32.04, 31.14, 30.35, 30.59, 30.78, 30.39, 31.42,
        35.68, 35.18, 35.11, 34.96, 35.43, 35.28, 34.82, 35.06, 35.71, 35.42, 35.23
    ],
    'ForcedExit2': [
        43.16, 48.97, 52.47, 59.05, 62.77, 67.39, 70.04, 73.01, 75.27, 79.01, 81.02,
        43.53, 49.47, 51.72, 58.60, 62.88, 66.84, 70.64, 72.97, 75.20, 79.21, 80.81,
        40.13, 46.67, 49.59, 56.01, 60.20, 64.53, 68.30, 71.86, 73.51, 78.01, 80.60,
        40.09, 45.25, 49.36, 55.17, 59.53, 65.11, 67.11, 71.53, 73.96, 78.34, 80.56
    ]
}

df = pd.DataFrame(data)

plt.figure(figsize=(14, 6))
plt.plot(df['Residual Block'], df['ForcedExit1'], marker='o', label='Forced Exit 1 Accuracy')
plt.plot(df['Residual Block'], df['ForcedExit2'], marker='o', label='Forced Exit 2 Accuracy')
plt.xticks(rotation=90)
plt.xlabel('Residual Block')
plt.ylabel('Accuracy (%)')
plt.title('No threshold exit accuracy for each trail')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
