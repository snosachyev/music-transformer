"""
Создание архитектуры Transformer (энкодер-декодер).
"""
from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
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


def build_triple_output_transformer(
        input_dim=3,  # pitch, step, duration
        d_model=256,
        num_heads=8,
        num_layers=4,
        ff_dim=512,
        dropout=0.1,
        seq_len=64
):
    # === Inputs ===
    enc_inputs = Input(shape=(seq_len, input_dim), name="encoder_input")
    dec_inputs = Input(shape=(None, input_dim), name="decoder_input")

    # === Linear projection ===
    enc_proj = Dense(d_model)(enc_inputs)
    dec_proj = Dense(d_model)(dec_inputs)

    # === Encoder ===
    for i in range(num_layers):
        attn = MultiHeadAttention(num_heads, d_model // num_heads)(enc_proj, enc_proj)
        enc_proj = LayerNormalization()(enc_proj + Dropout(dropout)(attn))
        ff = Dense(ff_dim, activation='relu')(enc_proj)
        ff = Dense(d_model)(ff)
        enc_proj = LayerNormalization()(enc_proj + Dropout(dropout)(ff))

    # === Decoder ===
    dec = dec_proj
    for i in range(num_layers):
        attn1 = MultiHeadAttention(num_heads, d_model // num_heads)(dec, dec)
        dec = LayerNormalization()(dec + Dropout(dropout)(attn1))

        attn2 = MultiHeadAttention(num_heads, d_model // num_heads)(dec, enc_proj)
        dec = LayerNormalization()(dec + Dropout(dropout)(attn2))

        ff = Dense(ff_dim, activation='relu')(dec)
        ff = Dense(d_model)(ff)
        dec = LayerNormalization()(dec + Dropout(dropout)(ff))

    # === Три головы (pitch, step, duration) ===
    pitch_out = Dense(1, name="pitch_out")(dec)
    step_out = Dense(1, activation='relu', name="step_out")(dec)
    dur_out = Dense(1, activation='relu', name="duration_out")(dec)

    # === Модель ===
    model = Model([enc_inputs, dec_inputs], [pitch_out, step_out, dur_out])
    model.compile(
        optimizer='adam',
        loss={'pitch_out': 'mse', 'step_out': 'mse', 'duration_out': 'mse'},
        loss_weights={'pitch_out': 1.0, 'step_out': 0.5, 'duration_out': 0.5}
    )
    return model
