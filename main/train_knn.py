import os
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import pickle
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf

model = tf.keras.models.load_model("facenet_custom_trained.keras")
dataset_path = "embeddings"

def preprocess(img_path):
    img = Image.open(img_path).resize((160, 160)).convert('RGB')
    img_array = np.asarray(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def load_dataset(dataset_path):
    X, y = [], []
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        for root, _, files in os.walk(person_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if file.endswith(".npy"):
                        data = np.load(file_path)
                        if data.shape == (128,):
                            X.append(data)
                            y.append(person_name)
                        else:
                            print(f"⚠️ Skipping {file_path}: shape {data.shape} is not (128,)")
                    elif file.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_array = preprocess(file_path)
                        embedding = model.predict(img_array, verbose=0)[0]
                        embedding = embedding.reshape(-1)
                        X.append(embedding)
                        y.append(person_name)
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")
    return np.array(X), np.array(y)

# load dataset recursively
X, y = load_dataset(dataset_path)

# train knn classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

os.makedirs("train", exist_ok=True)

with open("train/train.pkl", "wb") as f:
    pickle.dump((X, y), f)
    print("✅ Embeddings and labels saved to train/train.pkl")

with open("train/knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)
    print("✅ KNN model saved to train/knn_model.pkl")

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ KNN model accuracy: {accuracy * 100:.2f}%")
except Exception as e:
    print("⚠️ Error during evaluation:", e)

print("🎉 KNN training completed successfully!")
