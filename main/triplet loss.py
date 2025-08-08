import os
import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load 128-D embeddings dataset
# -----------------------------
def load_dataset(dataset_path):
    X, y = [], []
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir): continue

        for file in os.listdir(person_dir):
            if not file.endswith(".npy"): continue
            path = os.path.join(person_dir, file)
            try:
                data = np.load(path)
                data = np.asarray(data)
                if data.shape == (128,):
                    X.append(data)
                    y.append(person_name)
                else:
                    print(f"⚠️ Skipping {path}: shape {data.shape} is not (128,)")
            except Exception as e:
                print(f"❌ Error loading {path}: {e}")
    return np.array(X), np.array(y)

# -----------------------------
# Triplet loss function
# -----------------------------
def triplet_loss(y_true, y_pred, alpha=0.3):
    anchor, positive, negative = y_pred[:, 0:128], y_pred[:, 128:256], y_pred[:, 256:384]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    basic_loss = pos_dist - neg_dist + alpha
    return tf.reduce_mean(tf.maximum(basic_loss, 0.0))

# -----------------------------
# Generate (anchor, positive, negative) triplets
# -----------------------------
def generate_triplets(X, y):
    triplets = []
    for i in range(len(X)):
        anchor = X[i]
        label = y[i]

        pos_indices = np.where(y == label)[0]
        neg_indices = np.where(y != label)[0]

        if len(pos_indices) < 2 or len(neg_indices) == 0:
            continue

        positive = X[np.random.choice(pos_indices)]
        negative = X[np.random.choice(neg_indices)]

        triplets.append((anchor, positive, negative))
    return triplets

# -----------------------------
# MAIN TRAINING FLOW
# -----------------------------
if __name__ == "__main__":
    dataset_path = "embeddings"
    X, y = load_dataset(dataset_path)
    print(f"✅ Loaded {len(X)} valid embeddings.")

    le = LabelEncoder()
    y = le.fit_transform(y)

    triplets = generate_triplets(X, y)
    print(f"✅ Generated {len(triplets)} triplets.")

    if len(triplets) == 0:
        print("❌ No triplets generated. Check data.")
        exit()

    A = np.array([t[0] for t in triplets])
    P = np.array([t[1] for t in triplets])
    N = np.array([t[2] for t in triplets])
    y_dummy = np.zeros((len(A),))  # Required dummy output for triplet loss

    # -----------------------------
    # Triplet model using 128-D inputs
    # -----------------------------
    input_1 = tf.keras.Input(shape=(128,))
    input_2 = tf.keras.Input(shape=(128,))
    input_3 = tf.keras.Input(shape=(128,))
    merged = layers.concatenate([input_1, input_2, input_3], axis=1)

    model = models.Model(inputs=[input_1, input_2, input_3], outputs=merged)
    model.compile(optimizer='adam', loss=triplet_loss)

    print("🚀 Starting training...")
    model.fit([A, P, N], y_dummy, batch_size=4, epochs=10)

    model.save("triplet_trained_model.keras")
    print("✅ Triplet model saved as 'triplet_trained_model.keras'")
