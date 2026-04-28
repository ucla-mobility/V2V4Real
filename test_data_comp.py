import numpy as np
import matplotlib.pyplot as plt

frame = "0005"   # change frame number

base = "opencood/logs/v2xvit_opv2v_2026_03_09_18_15_32/npy"
gt = np.load(f"{base}/{frame}_gt.npy")
pred = np.load(f"{base}/{frame}_pred.npy")
pcd = np.load(f"{base}/{frame}_pcd.npy")

fig, axes = plt.subplots(1, 2, figsize=(14,7))

# -----------------
# LEFT: Ground Truth
# -----------------
axes[0].scatter(pcd[:,0], pcd[:,1], s=0.5, c='gray')

for box in gt:
    box = np.vstack((box, box[0]))  # close box
    axes[0].plot(box[:,0], box[:,1], c='green', linewidth=2)

axes[0].set_title("Ground Truth")
axes[0].axis("equal")

# -----------------
# RIGHT: Prediction
# -----------------
axes[1].scatter(pcd[:,0], pcd[:,1], s=0.5, c='gray')

for box in pred:
    box = np.vstack((box, box[0]))
    axes[1].plot(box[:,0], box[:,1], c='red', linewidth=2)

axes[1].set_title("Prediction")
axes[1].axis("equal")

# 👇 ADD THEM HERE
axes[0].set_xlim(-50,50)
axes[0].set_ylim(-50,50)

axes[1].set_xlim(-50,50)
axes[1].set_ylim(-50,50)

plt.savefig("comparison.png", dpi=300)