import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Data Extraction
# Columns: %Al2O3, aAl2O3, aMgO, aSiO2, aCaO
mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 9,
        "axes.labelsize": 10,  # 10-11 pt
        "axes.labelweight": "bold",  # bold
        "axes.titlesize": 11,
        "legend.fontsize": 8,  # Reduced
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 0.7,  # Default, overridden per panel
        "xtick.direction": "inout",
        "ytick.direction": "inout",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "svg.fonttype": "none",
        "mathtext.fontset": "stix",
    }
)
# %Al2O3 (Converted to % Mass)
pct_al2o3 = [
    x * 100 for x in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
]

# Activities
a_al2o3 = [
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

a_mgo = [
    0.088,
    0.071780822,
    0.06,
    0.054117647,
    0.048235294,
    0.042352941,
    0.035666667,
    0.025833333,
    0.019958824,
    0.01980625,
    0.019653846,
]

a_sio2 = [
    0.000546154,
    0.0010575,
    0.001566667,
    0.002,
    0.003034483,
    0.00405,
    0.005,
    0.006744186,
    0.008690476,
    0.01,
    0.014634146,
]

a_cao = [
    0.05,
    0.0375,
    0.029,
    0.0228125,
    0.017346939,
    0.013877551,
    0.01,
    0.007840909,
    0.005909091,
    0.003733333,
    0.002527273,
]


# Setup Plot
# 10.5 cm x 6 cm (approx 4.13 x 2.36 inches)
fig, ax1 = plt.subplots(figsize=(10.5 / 2.54, 6 / 2.54))

color1 = "tab:blue"
color2 = "tab:orange"
color3 = "tab:green"
color4 = "tab:red"

# Common line properties for "thick lines"
lw = 2.0  # Thicker lines
ms = 5  # Marker size

# Primary Axis - Higher Values
ax1.set_xlabel(r"% $Al_2O_3$ (%masa)")
ax1.set_ylabel(r"Actividad ($a_{Al_2O_3}$, $a_{MgO}$)", color="black")
(l1,) = ax1.plot(
    pct_al2o3,
    a_al2o3,
    marker="o",
    label=r"$a_{Al_2O_3}$",
    color=color1,
    linewidth=lw,
    markersize=ms,
)
(l2,) = ax1.plot(
    pct_al2o3,
    a_mgo,
    marker="s",
    label=r"$a_{MgO}$",
    color=color2,
    linewidth=lw,
    markersize=ms,
)
ax1.tick_params(axis="y", labelcolor="black")

# Secondary Axis - Lower Values
ax2 = ax1.twinx()
ax2.set_ylabel(r"Actividad ($a_{SiO_2}$, $a_{CaO}$)", color="black")
(l3,) = ax2.plot(
    pct_al2o3,
    a_sio2,
    marker="^",
    label=r"$a_{SiO_2}$",
    color=color3,
    linestyle="--",
    linewidth=lw,
    markersize=ms,
)
(l4,) = ax2.plot(
    pct_al2o3,
    a_cao,
    marker="D",
    label=r"$a_{CaO}$",
    color=color4,
    linestyle="--",
    linewidth=lw,
    markersize=ms,
)
ax2.tick_params(axis="y", labelcolor="black")

# Combined Legend
# Combined Legend
lines = [l1, l2, l3, l4]
labels = [l.get_label() for l in lines]
ax1.legend(
    lines,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=4,
    frameon=False,
    columnspacing=1.0,
    handletextpad=0.4,
)



plt.subplots_adjust(top=0.85, bottom=0.18, left=0.15, right=0.85)
plt.grid(True)
# plt.tight_layout() # Conflict with subplots_adjust
plt.show()
