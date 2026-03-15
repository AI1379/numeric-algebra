import numpy as np
import matplotlib.pyplot as plt

# Using PGF backend configuration for maximum LaTeX integration
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,          # Handle fonts via LaTeX
    "pgf.texsystem": "lualatex",   # Matches your document engine
    "font.size": 10,               # Default LaTeX size
    "pgf.preamble": "\n".join([
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
    ])
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

# Define the grid in the z-plane
# x in [-pi/2, 3pi/2], y in [0, 2] is a good region to show the mapping
x = np.linspace(-np.pi, np.pi, 13)  # x lines (vertical)
y = np.linspace(0, 2.5, 9)          # y lines (horizontal)
X, Y = np.meshgrid(np.linspace(-np.pi, np.pi, 200), np.linspace(0, 2.5, 200))
Z_dense = X + 1j * Y

# ==================== Left Plot (z-plane) ====================
# Draw constant x lines (vertical)
# We will use distinct colors to track them
colors_x = plt.cm.viridis(np.linspace(0, 1, len(x)))
for i, xi in enumerate(x):
    z_line = xi + 1j * np.linspace(0, 2.5, 200)
    ax1.plot(z_line.real, z_line.imag, color=colors_x[i], lw=1.5, alpha=0.8)

# Draw constant y lines (horizontal)
colors_y = plt.cm.plasma(np.linspace(0, 1, len(y)))
for i, yi in enumerate(y):
    z_line = np.linspace(-np.pi, np.pi, 200) + 1j * yi
    ax1.plot(z_line.real, z_line.imag, color=colors_y[i], linestyle='--', lw=1.5, alpha=0.8)

# Add some visual markers (like the T-shape) to show orientation
# Let's draw a small 'L' shape
L_mask = ((X >= 0.2) & (X <= 0.8) & (Y >= 0.5) & (Y <= 0.8)) | \
         ((X >= 0.2) & (X <= 0.4) & (Y >= 0.8) & (Y <= 1.5))
ax1.plot(X[L_mask], Y[L_mask], 'k.', markersize=0.5, alpha=0.9)

ax1.set_xlim(-np.pi - 0.2, np.pi + 0.2)
ax1.set_ylim(-0.2, 2.7)
ax1.set_aspect('equal')
ax1.set_title(r"z-plane: $z = x + iy$")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$y$")
ax1.grid(True, alpha=0.3)

# Add ticks for pi
ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax1.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])


# ==================== Right Plot (w-plane) ====================
# Mapping: w = cos(z)
# Draw mapped constant x lines (Hyperbolas)
for i, xi in enumerate(x):
    z_line = xi + 1j * np.linspace(0, 2.5, 200)
    w_line = np.cos(z_line)
    ax2.plot(w_line.real, w_line.imag, color=colors_x[i], lw=1.5, alpha=0.8)

# Draw mapped constant y lines (Ellipses)
for i, yi in enumerate(y):
    z_line = np.linspace(-np.pi, np.pi, 200) + 1j * yi
    w_line = np.cos(z_line)
    ax2.plot(w_line.real, w_line.imag, color=colors_y[i], linestyle='--', lw=1.5, alpha=0.8)

# Map the 'L' shape
W_dense = np.cos(Z_dense)
ax2.plot(W_dense[L_mask].real, W_dense[L_mask].imag, 'k.', markersize=0.5, alpha=0.9)

# Mark the foci of the ellipses and hyperbolas at -1 and 1
ax2.plot([-1, 1], [0, 0], 'ro', markersize=5, label='Foci ($\\pm 1$)')
ax2.legend()

ax2.set_xlim(-4, 4)
ax2.set_ylim(-3, 3)
ax2.set_aspect('equal')
ax2.set_title(r"w-plane: $w = \cos(z)$")
ax2.set_xlabel(r"$u$")
ax2.set_ylabel(r"$v$")
ax2.grid(True, alpha=0.3)

# plt.tight_layout()

# Save as PGF for native LaTeX integration
output_pgf = "mapping_cos.pgf"
plt.savefig(output_pgf, format="pgf")
print(f"Plot saved to {output_pgf}")
