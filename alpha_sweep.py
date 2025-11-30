import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import yaml
from sklearn.metrics import pairwise_distances

ENV = yaml.load(open("environment.yaml", "r"), Loader=yaml.FullLoader)

n_samples = ENV["n_samples"]
cluster_type = ENV["cluster_type"]

a = np.ones(n_samples) / n_samples
b = np.ones(n_samples) / n_samples

print("Loading data")
# Load the data
if cluster_type == "kmeans":
    X_sad_audio = np.load("./data/X_sad_audio.npy")
    X_joy_audio = np.load("./data/X_joy_audio.npy")

    X_sad_text = np.load("./data/X_sad_text.npy")
    X_joy_text = np.load("./data/X_joy_text.npy")

    X_sad_text_ref = np.load("./data/X_sad_text_ref.npy", allow_pickle=True)
    X_joy_text_ref = np.load("./data/X_joy_text_ref.npy", allow_pickle=True)
else:
    # If you set it to anything else, it will use the given clusters from the data
    X_sad_audio = np.load("./data/X_sad_audio_2.npy")
    X_joy_audio = np.load("./data/X_joy_audio.npy")

    X_sad_text = np.load("./data/X_sad_text_2.npy")
    X_joy_text = np.load("./data/X_joy_text_2.npy")

    X_sad_text_ref = np.load("./data/X_sad_text_ref_2.npy", allow_pickle=True)
    X_joy_text_ref = np.load("./data/X_joy_text_ref_2.npy", allow_pickle=True)

# Compute pairwise distances for ground metrics
C_audio = pairwise_distances(X_sad_audio, X_joy_audio, metric="euclidean")
C_text  = pairwise_distances(X_sad_text,  X_joy_text,  metric="euclidean")

# Normalize by max value to put them on similar scales (0 to 1)
C_audio_norm = C_audio / C_audio.max()
C_text_norm  = C_text  / C_text.max()

np.save("./data/C_audio_norm.npy", C_audio_norm)
np.save("./data/C_text_norm.npy",  C_text_norm)

print("Saved normalized cost matrices:")
print("./data/C_audio_norm.npy")
print("./data/C_text_norm.npy")

print("Alpha Sweep")
# 2. Alpha Sweep Experiment
alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
acoustic_dists = []
semantic_dists = []

for alpha in alphas:
    # Hybrid Cost
    C_curr = alpha * C_audio_norm + (1 - alpha) * C_text_norm

    # Sinkhorn
    T_curr = ot.sinkhorn(a, b, C_curr, reg=0.01)

    # Acoustic Outcome: Project Sad Audio -> Joy Space
    mapped_audio = T_curr.dot(X_joy_audio) * n_samples

    # Audio Metric
    K = 3  # number of nearest neighbors to average over

    # Compute full pairwise distances between mapped audio and Joy audio
    dists = np.linalg.norm(
        mapped_audio[:, None, :] - X_joy_audio[None, :, :],
        axis=2
    )

    # Sort each row: nearest to farthest Joy samples
    sorted_dists = np.sort(dists, axis=1)

    # Take the mean of the K nearest Joy distances
    knn_mean = sorted_dists[:, :K].mean(axis=1)

    # Final acoustic metric: average over all sad samples
    d_acoustic = knn_mean.mean()

    acoustic_dists.append(d_acoustic)

    # Semantic Outcome
    mapped_text = T_curr.dot(X_joy_text) * n_samples
    d_semantic = np.linalg.norm(X_sad_text - mapped_text, axis=1).mean()
    semantic_dists.append(d_semantic)

print("Sweep complete")

# Analysis
acc_arr = np.array(acoustic_dists)
sem_arr = np.array(semantic_dists)

acc_norm = (acc_arr - acc_arr.min()) / (acc_arr.max() - acc_arr.min())
sem_norm = (sem_arr - sem_arr.min()) / (sem_arr.max() - sem_arr.min())
combined_score = acc_norm + sem_norm

best_idx = np.argmin(combined_score)
best_alpha = alphas[best_idx]

results_df = pd.DataFrame({
    "Alpha": alphas,
    "Acoustic Dist": acc_arr,
    "Semantic Dist": sem_arr,
    "Combined Score": combined_score
})

print("Saving results to alpha_sweep_results.csv")
results_df.to_csv("./data/alpha_sweep_results.csv", index=False)
print(f"Based on equal weighting of normalized scores, the best Alpha is: {best_alpha}")


# Plot the data
plt.figure(figsize=(10, 6))

# Plot Normalized Acoustic Distance (Lower is better emotion match)
plt.plot(alphas, acc_norm, marker='o', label='Norm. Acoustic Dist (Emotion Error)', color='blue')

# Plot Normalized Semantic Distance (Lower is better content preservation)
plt.plot(alphas, sem_norm, marker='s', label='Norm. Semantic Dist (Content Loss)', color='orange')

# Plot Combined Score (Lower is better)
plt.plot(alphas, combined_score, marker='D', linestyle='--', linewidth=2, label='Combined Score', color='green')

# Highlight the optimal alpha
best_alpha_idx = np.argmin(combined_score)
best_alpha_val = alphas[best_alpha_idx]
plt.axvline(x=best_alpha_val, color='gray', linestyle=':', label=f'Optimal Alpha: {best_alpha_val}')

plt.title("Alpha Sweep: Balancing Emotion vs. Content")
plt.xlabel("Alpha (Weight of Acoustic Cost)")
plt.ylabel("Normalized Distance / Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()