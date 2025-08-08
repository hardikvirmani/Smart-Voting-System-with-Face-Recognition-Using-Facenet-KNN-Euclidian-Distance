import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionResNetV2

# -----------------------------
# Build a FaceNet-style embedding model
# -----------------------------
def build_facenet_model():
    base_model = InceptionResNetV2(
        input_shape=(160, 160, 3),
        include_top=False,
        pooling='avg',
        weights='imagenet'
    )

    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = base_model(inputs, training=False)
    x = layers.Dense(128)(x)
    outputs = tf.nn.l2_normalize(x, axis=1)

    model = models.Model(inputs, outputs)
    return model

# -----------------------------
# Build and save the model
# -----------------------------
model = build_facenet_model()

# Save in modern format (.keras)
model.save("facenet_custom_trained.keras")
print("✅ Saved model as facenet_custom_trained.keras")

# Save in legacy format (.h5)
model.save("facenet_custom_trained.h5")
print("✅ Also saved model as facenet_custom_trained.h5 (HDF5 legacy)")
