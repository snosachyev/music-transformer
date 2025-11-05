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
