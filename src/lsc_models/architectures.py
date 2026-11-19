"""BiLSTM y Transformer compacto sobre la entrada contractual ``(6, 147)``.

Las dos arquitecturas comparten: entrada, proyección lineal por instante a un
mismo ancho, regularización por dropout, salida softmax de 10 clases y el mismo
presupuesto de entrenamiento (ver ``training`` en la configuración). Los números
de parámetros se registran en cada ejecución para que la comparación sea
auditable; no se afinan hiperparámetros fuera de la configuración declarada.

TensorFlow se importa de forma perezosa.
"""

from __future__ import annotations

from typing import Any

SEQ_LEN = 6
FEAT_DIM = 147
NUM_CLASSES = 10


def build_bilstm(params: dict) -> Any:
    import keras
    from keras import layers

    units = int(params["units"])
    proj = int(params["proj_dim"])
    dropout = float(params["dropout"])
    inp = keras.Input(shape=(SEQ_LEN, FEAT_DIM), dtype="float32", name="features")
    x = layers.Dense(proj, activation="relu", name="frame_proj")(inp)
    x = layers.Dropout(dropout, name="drop_in")(x)
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=False), name="bilstm")(x)
    x = layers.Dropout(dropout, name="drop_out")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="probs")(x)
    return keras.Model(inp, out, name="bilstm")


def build_transformer(params: dict) -> Any:
    import keras
    from keras import layers

    from .custom_layers import PositionalEmbedding

    d_model = int(params["d_model"])
    heads = int(params["num_heads"])
    blocks = int(params["num_blocks"])
    ff = int(params["ff_dim"])
    dropout = float(params["dropout"])

    inp = keras.Input(shape=(SEQ_LEN, FEAT_DIM), dtype="float32", name="features")
    x = layers.Dense(d_model, name="frame_proj")(inp)
    x = PositionalEmbedding(SEQ_LEN, d_model, name="pos")(x)
    x = layers.Dropout(dropout, name="drop_in")(x)
    for i in range(blocks):
        h = layers.LayerNormalization(epsilon=1e-6, name=f"ln_att_{i}")(x)
        h = layers.MultiHeadAttention(
            num_heads=heads, key_dim=d_model // heads, dropout=dropout, name=f"mha_{i}"
        )(h, h)
        x = layers.Add(name=f"res_att_{i}")([x, h])
        h = layers.LayerNormalization(epsilon=1e-6, name=f"ln_ff_{i}")(x)
        h = layers.Dense(ff, activation="relu", name=f"ff1_{i}")(h)
        h = layers.Dropout(dropout, name=f"ff_drop_{i}")(h)
        h = layers.Dense(d_model, name=f"ff2_{i}")(h)
        x = layers.Add(name=f"res_ff_{i}")([x, h])
    x = layers.LayerNormalization(epsilon=1e-6, name="ln_out")(x)
    x = layers.GlobalAveragePooling1D(name="pool")(x)
    x = layers.Dropout(dropout, name="drop_out")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="probs")(x)
    return keras.Model(inp, out, name="transformer")


BUILDERS = {"bilstm": build_bilstm, "transformer": build_transformer}


def build_model(kind: str, params: dict) -> Any:
    if kind not in BUILDERS:
        raise ValueError(f"arquitectura desconocida: {kind}")
    return BUILDERS[kind](params)
