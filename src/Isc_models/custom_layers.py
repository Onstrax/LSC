"""Capas personalizadas serializables (necesarias para recargar ``model.keras``).

Se importa SIEMPRE antes de ``keras.models.load_model`` (Predictor, sensibilidad).
"""

from __future__ import annotations

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="lsc")
class PositionalEmbedding(layers.Layer):
    """Embedding posicional aprendido para los instantes de la secuencia."""

    def __init__(self, length: int, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.length = int(length)
        self.dim = int(dim)
        self.emb = layers.Embedding(self.length, self.dim)

    def build(self, input_shape):
        self.emb.build((None,))
        super().build(input_shape)

    def call(self, x):
        return x + self.emb(ops.arange(self.length))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"length": self.length, "dim": self.dim})
        return cfg
