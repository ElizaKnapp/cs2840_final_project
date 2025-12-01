import numpy as np
import ot
import umap
import yaml
import matplotlib.pyplot as plt

ENV = yaml.load(open("environment.yaml", "r"), Loader=yaml.FullLoader)

n_samples = ENV["n_samples"]
cluster_type = ENV["cluster_type"]

np.random.seed(1)

def get_alpha():
    while True:
        try:
            alpha = float(input("Enter alpha (0 to 1): "))
            if 0.0 <= alpha <= 1.0:
                return alpha
            else:
                print("Alpha must be between 0 and 1.")
        except ValueError:
            print("Please enter a valid float.")

alpha = get_alpha()
print("Using alpha =", alpha)

# Hybrid Cost
C_audio_norm = np.load("./data/C_audio_norm.npy")
C_text_norm  = np.load("./data/C_text_norm.npy")
C_final = alpha * C_audio_norm + (1 - alpha) * C_text_norm

# Import the audio/text/text references
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

# Sinkhorn
a = np.ones(n_samples) / n_samples
b = np.ones(n_samples) / n_samples
T_final = ot.sinkhorn(a, b, C_final, reg=0.01)

# 1. Acoustic Outcome: Project Sad Audio -> Joy Space
mapped_audio = T_final.dot(X_joy_audio) * n_samples
mapped_text  = T_final.dot(X_joy_text) * n_samples

# Stack the arrays to fit UMAP on the joint space
# Order: Sad (Original), Joy (Target), Sad (Mapped Hybrid)
X_all_audio = np.vstack([X_sad_audio, X_joy_audio, mapped_audio])
X_all_text  = np.vstack([X_sad_text,  X_joy_text,  mapped_text])

mapped_joint = np.hstack([mapped_audio, mapped_text])
sad_joint = np.hstack([X_sad_audio, X_sad_text])
joy_joint = np.hstack([X_joy_audio, X_joy_text])

X_all = np.vstack([sad_joint, joy_joint, mapped_joint])

# Initialize and fit UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
Z_all = reducer.fit_transform(X_all)

# Split the transformed data back into groups
Z_sad = Z_all[:n_samples]
Z_joy = Z_all[n_samples:2*n_samples]
Z_mapped = Z_all[2*n_samples:]

## Reconstruct Text

def find_nn(X, Y):
  dists = np.sum((Y - X)**2, axis=1)
  idx = np.argmin(dists)
  return Y[idx], idx

nn_arr = []
for i in range(300):
  sample = np.random.randint(low=0, high=len(X_joy_audio))
  mapped_audio_s = Z_mapped[sample]
  text = X_sad_text_ref[sample]
  # find nearest neighbor to mapped audio from within X_sad_audio
  nn_audio, nearest_neighbor = find_nn(mapped_audio_s, Z_joy)
  nn_arr.append(nearest_neighbor)
  # index of NN
  nn_text = X_joy_text_ref[nearest_neighbor]
arr_n, bins = np.histogram(nn_arr, bins=range(n_samples))

non_empty_bins = arr_n[arr_n > 0]
# Count non-empty bins
num_non_empty_bins = len(non_empty_bins)

# Calculate total number of bins
total_bins = len(bins) - 1

# Calculate the ratio
ratio = num_non_empty_bins / total_bins

print(f"Number of non-empty bins: {num_non_empty_bins}")
print(f"Total number of bins: {total_bins}")
print(f"Ratio of non-empty bins to total bins: {ratio:.2f}")

avg_points_per_nonempty_bin = non_empty_bins.mean()
print(f"Average points per non-empty bin: {avg_points_per_nonempty_bin:.2f}")


# Top k matches
def top_k_ot_matches(T, X_sad_text_ref, X_joy_text_ref, K=3):
    """
    Extract the top-K strongest OT matchings globally.
    """
    flat = T.flatten()
    top_idx = np.argsort(flat)[::-1][:K]

    print("Sanity checking text mapping")
    results = []

    for rank, idx in enumerate(top_idx):
        i = idx // T.shape[1]   # sad index
        j = idx % T.shape[1]    # joy index
        mass = T[i, j]

        sad_text = X_sad_text_ref[i]
        joy_text = X_joy_text_ref[j]

        results.append((i, j, mass, sad_text, joy_text))

        print(f"Rank {rank+1} (mass={mass:.6f})")
        print(f"SAD[{i}] -> JOY[{j}]")
        print(f"  SAD text : {sad_text}")
        print(f"  JOY text : {joy_text}")
        print()

    return results

# Run the top-K OT match extraction
top_ot_results = top_k_ot_matches(T_final, X_sad_text_ref, X_joy_text_ref, K=3)

## All of the graphs

fig, axs = plt.subplots(1, 3, figsize=(24, 7))

# Full umap with all points
axs[0].scatter(Z_sad[:,0], Z_sad[:,1], c='blue', s=25, alpha=0.5, label="Sad")
axs[0].scatter(Z_joy[:,0], Z_joy[:,1], c='orange', s=25, alpha=0.5, label="Joy")
axs[0].scatter(Z_mapped[:,0], Z_mapped[:,1], c='green', alpha=0.5, marker='x', s=35, label="Mapped")
axs[0].set_title(f"Joint Multimodal OT Mapping\nalpha={alpha}", fontsize=14)
axs[0].legend()
axs[0].grid(alpha=0.3)

# Histogram of NN hits
axs[1].hist(nn_arr, bins=range(n_samples), color='purple', alpha=0.7)
axs[1].set_title("NN Distribution (Mapped to Joy Cluster)", fontsize=14)
axs[1].set_xlabel("Joy Index")
axs[1].set_ylabel("Frequency")

# Filtered umap with only joy nn points
nn_points = Z_joy[nn_arr]
axs[2].scatter(Z_sad[:,0], Z_sad[:,1], c='blue', s=25, alpha=0.5, label="Sad")
axs[2].scatter(nn_points[:,0], nn_points[:,1], c='orange', s=35, alpha=0.5, label="Joy NN Only")
axs[2].scatter(Z_mapped[:,0], Z_mapped[:,1], c='green', alpha=0.5, marker='x', s=35, label="Mapped")
axs[2].set_title("Mapped Points vs Joy Nearest Neighbors", fontsize=14)
axs[2].legend()
axs[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()