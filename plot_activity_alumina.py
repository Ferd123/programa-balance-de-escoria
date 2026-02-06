import matplotlib.pyplot as plt
import numpy as np

# Data from user request
# Columns: C/A, %CaO, %SiO2, %MgO, %Al2O3, B, aAl2O3, ...
# We need %Al2O3 (4th index if 0-based from data lines provided) and aAl2O3 (6th index)

# %Al2O3
pct_al2o3 = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]

# aAl2O3
activity_al2o3 = [
    0.0012,
    0.00443662,
    0.008518519,
    0.0125,
    0.018571429,
    0.025681818,
    0.036,
    0.047377049,
    0.067,
    0.088829787,
    0.123571429,
]

# Convert to numpy arrays for easier handling if needed later
x = np.array(pct_al2o3)
y = np.array(activity_al2o3)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker="o", linestyle="-", color="b", label="Activity of Alumina")

# Labels and Title
plt.xlabel("% Al2O3 (Alumina Mass Fraction)")
plt.ylabel("aAl2O3 (Activity of Alumina)")
plt.title("Activity vs % Alumina")
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
