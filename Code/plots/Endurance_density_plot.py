import matplotlib.pyplot as plt
import numpy as np

# Memory data
memory_types = ["SRAM", "ReRAM", "PCM", "FeFET", "STT-MRAM", "SOT-MRAM", "Flash"]
endurance = [1e16, 1e9, 1e8, 1e5, 1e15, 1e15, 1e4]
density = [1e6, 1.6e8, 1e8, 5e7, 1e7, 2e7, 3e8]  # bits/mm²

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(density, endurance, c='blue')

# Annotate each point
for i, mem in enumerate(memory_types):
    plt.annotate(mem, (density[i]*1.05, endurance[i]*1.05), fontsize=9)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Density (bits/mm², log scale)")
plt.ylabel("Write Endurance (cycles, log scale)")
plt.title("Write Endurance vs Density for Emerging Memory Technologies")
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()
