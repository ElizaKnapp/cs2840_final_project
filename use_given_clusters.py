import pandas as pd
import pickle
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score

ENV = yaml.load(open("environment.yaml", "r"), Loader=yaml.FullLoader)
n_samples = ENV["n_samples"]

print("Retriving data")
# Get the data
train_df = pd.read_csv("./data/train_sent_emo.csv")
dev_df   = pd.read_csv("./data/dev_sent_emo.csv")
test_df  = pd.read_csv("./data/test_sent_emo.csv")

with open("./data/audio_emotion.pkl", "rb") as f:
    train_audio_emb, val_audio_emb, test_audio_emb = pickle.load(f)

with open("./data/text_emotion.pkl", "rb") as f:
    train_text_emb, val_text_emb, test_text_emb = pickle.load(f)

def pickle_dialog_dict_to_df(emb_dict, split_name, vec_col_name):
    """
    emb_dict: e.g., train_audio_emb (dict[str -> np.ndarray[num_utts, dim]])
    split_name: "train" / "dev" / "test"
    vec_col_name: e.g., "audio_vec"
    """
    rows = []
    for dia_str, mat in emb_dict.items():
        dia_id = int(dia_str)
        num_utts, _ = mat.shape
        for utt_id in range(num_utts):
            rows.append({
                "Dialogue_ID": dia_id,
                "Utterance_ID": utt_id,
                "split": split_name,
                vec_col_name: mat[utt_id],
            })
    return pd.DataFrame(rows)

train_audio_table = pickle_dialog_dict_to_df(train_audio_emb, "train", "audio_vec")
dev_audio_table   = pickle_dialog_dict_to_df(val_audio_emb,   "dev",   "audio_vec")
test_audio_table  = pickle_dialog_dict_to_df(test_audio_emb,  "test",  "audio_vec")

print("Merging data")
# Merge the data
train_merged = pd.merge(
    train_df,
    train_audio_table,
    on=["Dialogue_ID", "Utterance_ID"],
    how="inner"
)

dev_merged = pd.merge(
    dev_df,
    dev_audio_table,
    on=["Dialogue_ID", "Utterance_ID"],
    how="inner"
)

test_merged = pd.merge(
    test_df,
    test_audio_table,
    on=["Dialogue_ID", "Utterance_ID"],
    how="inner"
)

# Turn list of arrays into a 2D matrix
X_train = np.stack(train_merged["audio_vec"].to_numpy())  # shape: [N_train, dim]

# Optional: also build label vector for evaluation ONLY (not used in clustering)
emotions = sorted(train_merged["Emotion"].unique())
emotion_to_id = {e: i for i, e in enumerate(emotions)}
y_train = train_merged["Emotion"].map(emotion_to_id).to_numpy()
cluster_map = {s: i for i, s in enumerate(emotions)}

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("Clustering data")

# Attach cluster IDs back to the DataFrame from given
train_merged["cluster"] = train_merged["Emotion"].map(cluster_map)
cluster_ids = train_merged["cluster"]

# Normalized Mutual Information between true emotion and cluster
nmi = normalized_mutual_info_score(y_train, train_merged["cluster"])

# Contingency table: how emotions distribute across clusters
ct = pd.crosstab(train_merged["cluster"], train_merged["Emotion"])

# Crosstab: cluster vs emotion
cluster_emotion = pd.crosstab(train_merged["cluster"], train_merged["Emotion"])
# Pick the cluster with most sadness and most joy
sad_cluster = (cluster_emotion["sadness"] / cluster_emotion.sum(axis=1)).idxmax()
joy_cluster = (cluster_emotion["joy"] / cluster_emotion.sum(axis=1)).idxmax()

X_sad = X_train_scaled[cluster_ids == sad_cluster]
X_joy = X_train_scaled[cluster_ids == joy_cluster]

# Sample 200 from the joy and sadness dominanted clusters
np.random.seed(42)

sad_sample = X_sad[np.random.choice(len(X_sad), n_samples, replace=False)]
joy_sample = X_joy[np.random.choice(len(X_joy), n_samples, replace=False)]

# Add the text data
train_text_table = pickle_dialog_dict_to_df(train_text_emb, "train", "text_vec")

print("Combining audio with text")
# Merge with the already merged audio/metadata dataframe
train_aligned = pd.merge(
    train_merged,
    train_text_table,
    on=["Dialogue_ID", "Utterance_ID"],
    how="inner"
)

# Filter for the identified Sad (0) and Joy (1) clusters
sad_aligned_df = train_aligned[train_aligned["cluster"] == sad_cluster]
joy_aligned_df = train_aligned[train_aligned["cluster"] == joy_cluster]

# Sample 200 instances from each, using a fixed seed for reproducibility
sad_sample_df = sad_aligned_df.sample(n=n_samples, random_state=42)
joy_sample_df = joy_aligned_df.sample(n=n_samples, random_state=42)

# Extract embeddings into numpy arrays
X_sad_audio = np.stack(sad_sample_df["audio_vec"].to_numpy())
X_joy_audio = np.stack(joy_sample_df["audio_vec"].to_numpy())

X_sad_text = np.stack(sad_sample_df["text_vec"].to_numpy())
X_joy_text = np.stack(joy_sample_df["text_vec"].to_numpy())

X_sad_text_ref = np.stack(sad_sample_df["Utterance"])
X_joy_text_ref = np.stack(joy_sample_df["Utterance"])

output_dir = "./data/"
arrays_to_save = {
    "X_sad_audio_2": X_sad_audio,
    "X_joy_audio_2": X_joy_audio,
    "X_sad_text_2": X_sad_text,
    "X_joy_text_2": X_joy_text,
    "X_sad_text_ref_2": X_sad_text_ref,
    "X_joy_text_ref_2": X_joy_text_ref,
}

for name, arr in arrays_to_save.items():
    path = f"{output_dir}{name}.npy"
    np.save(path, arr)
    print(f"Saved: {path}")
