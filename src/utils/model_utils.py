"""
Создание архитектуры Transformer (энкодер-декодер).
"""
from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, Activation

from keras.models import Model


def transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout=0.1):
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn = Dropout(dropout)(attn)
    x = LayerNormalization(epsilon=1e-6)(x + attn)
    ff = Dense(ff_dim, activation='relu')(x)
    ff = Dense(d_model)(ff)
    ff = Dropout(dropout)(ff)
    return LayerNormalization(epsilon=1e-6)(x + ff)


def transformer_decoder_block(x, enc, d_model, num_heads, ff_dim, dropout=0.1):
    self_attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    self_attn = Dropout(dropout)(self_attn)
    x = LayerNormalization(epsilon=1e-6)(x + self_attn)
    cross_attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, enc)
    cross_attn = Dropout(dropout)(cross_attn)
    x = LayerNormalization(epsilon=1e-6)(x + cross_attn)
    ff = Dense(ff_dim, activation='relu')(x)
    ff = Dense(d_model)(ff)
    ff = Dropout(dropout)(ff)
    return LayerNormalization(epsilon=1e-6)(x + ff)


def build_transformer(F=3, d_model=128, num_heads=4, ff_dim=256, num_layers=2, dropout=0.1):
    """
    Создает простую Transformer seq2seq модель.
    """
    enc_inputs = Input(shape=(None, F))
    dec_inputs = Input(shape=(None, F))
    x_enc = Dense(d_model)(enc_inputs)
    for _ in range(num_layers):
        x_enc = transformer_encoder_block(x_enc, d_model, num_heads, ff_dim, dropout)
    x_dec = Dense(d_model)(dec_inputs)
    for _ in range(num_layers):
        x_dec = transformer_decoder_block(x_dec, x_enc, d_model, num_heads, ff_dim, dropout)
    outputs = Dense(F)(x_dec)
    model = Model([enc_inputs, dec_inputs], outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


def build_triple_output_transformer(d_model=192, num_heads=4, ff_dim=512,
                                    enc_feat=3, dec_feat=3, num_pitch_classes=88,
                                    step_activation='relu', dur_activation='relu'):
    """
    Возвращает Keras Model:
      inputs: [enc_inputs, dec_inputs]
      outputs: [pitch_logits (no activation), step_out, duration_out]
    step_out и duration_out имеют активацию step_activation и dur_activation (relu по умолчанию).
    """
    enc_inputs = Input(shape=(None, enc_feat), name="encoder_inputs")
    dec_inputs = Input(shape=(None, dec_feat), name="decoder_inputs")

    # проекция
    enc_proj = Dense(d_model, name="enc_proj")(enc_inputs)
    dec_proj = Dense(d_model, name="dec_proj")(dec_inputs)

    # Encoder stack (маленький)
    x = enc_proj
    for i in range(3):
        att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, name=f"enc_mha_{i}")(x, x)
        x = LayerNormalization(name=f"enc_ln1_{i}")(Add()([x, att]))
        f = Dense(ff_dim, activation="relu", name=f"enc_ff1_{i}")(x)
        f = Dense(d_model, name=f"enc_ff2_{i}")(f)
        x = LayerNormalization(name=f"enc_ln2_{i}")(Add()([x, f]))
    enc_out = x  # (batch, L_enc, d_model)

    # Decoder stack
    y = dec_proj
    for i in range(3):
        att1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, name=f"dec_self_mha_{i}")(y, y)
        y = LayerNormalization(name=f"dec_ln_self_{i}")(Add()([y, att1]))

        att2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, name=f"dec_cross_mha_{i}")(y,
                                                                                                                enc_out)
        y = LayerNormalization(name=f"dec_ln_cross_{i}")(Add()([y, att2]))

        f = Dense(ff_dim, activation="relu", name=f"dec_ff1_{i}")(y)
        f = Dense(d_model, name=f"dec_ff2_{i}")(f)
        y = LayerNormalization(name=f"dec_ln_ff_{i}")(Add()([y, f]))

    # outputs
    # pitch — logits (softmax будет внутри loss)
    pitch_out = Dense(num_pitch_classes, activation=None, name="pitch_out")(y)

    # step/dur — используем relu (или указанную активацию), чтобы гарантировать >=0
    step_out = Dense(1, activation=step_activation, name="step_out")(y)
    dur_out = Dense(1, activation=dur_activation, name="duration_out")(y)

    model = Model([enc_inputs, dec_inputs], [pitch_out, step_out, dur_out])
    return model
