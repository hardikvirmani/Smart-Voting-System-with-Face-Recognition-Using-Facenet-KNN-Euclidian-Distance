import tensorflow as tf 
from keras.layers import Layer
from keras import backend as K
from keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
class CustomScaleLayer(Layer):
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = tf.Variable(
            initial_value=float(scale),
            trainable=True,
            dtype=tf.float32,
            name='scale'
        )

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return [x * self.scale for x in inputs]
        return inputs * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({'scale': float(self.scale.numpy())})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="Custom")
class L2Normalization(Layer):
    def __init__(self, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
