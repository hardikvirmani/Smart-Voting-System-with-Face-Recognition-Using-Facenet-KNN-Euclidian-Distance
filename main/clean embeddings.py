import os
import numpy as np

def fix_batch_embeddings(directory="embeddings"):
    for person in os.listdir(directory):
        person_dir = os.path.join(directory, person)
        if not os.path.isdir(person_dir):
            continue

        for file in os.listdir(person_dir):
            path = os.path.join(person_dir, file)

            if file.endswith(".npy"):
                try:
                    arr = np.load(path)
                    if arr.shape == (160, 160, 3):
                        print(f"❌ Removing raw image saved as .npy: {path}")
                        os.remove(path)
                    elif len(arr.shape) == 2 and arr.shape[1] == 128:
                        print(f"🔄 Splitting batch embedding: {path}")
                        for i, emb in enumerate(arr):
                            np.save(os.path.join(person_dir, f"{file[:-4]}_split_{i}.npy"), emb)
                        os.remove(path)
                except Exception as e:
                    print(f"⚠️ Failed to load {path}: {e}")

fix_batch_embeddings()
