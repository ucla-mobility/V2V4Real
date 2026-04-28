import numpy as np
import matplotlib.pyplot as plt

frame = "0951"   # change frame number

base = "opencood/logs/point_pillar_fcooper_2026_04_14_21_23_41/npy"
gt = np.load(f"{base}/{frame}_gt.npy")
pred = np.load(f"{base}/{frame}_pred.npy")
pcd = np.load(f"{base}/{frame}_pcd.npy")

plt.figure(figsize=(8,8))

plt.scatter(pcd[:,0], pcd[:,1], s=0.5, c='gray')

# GT boxes (yellow)
for box in gt:
    box = np.vstack((box, box[0]))
    plt.plot(box[:,0], box[:,1], c='green', linewidth=3)

# predicted boxes (red)
for box in pred:
    box = np.vstack((box, box[0]))
    plt.plot(box[:,0], box[:,1], c='red', linestyle='--', linewidth=2)

plt.title("Green = GT | Red = Prediction")
plt.axis("equal")

plt.savefig("fcooper", dpi=300)