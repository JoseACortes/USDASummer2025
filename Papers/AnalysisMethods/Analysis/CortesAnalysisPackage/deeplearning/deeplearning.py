import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input

def deep_learning_predict(df, existence, mixed_ratios, peak_areas, epochs=100, batch_size=16):
    """
    Build and train a deep learning model for spectrum analysis.
    Args:
        df: DataFrame of measured spectra (shape: [num_samples, 935])
        existence: binary existence array (num_samples, 8)
        mixed_ratios: array of mixed ratios (num_samples, 8)
        peak_areas: array of peak areas (num_samples, 1)
    Returns:
        model: trained Keras model
        history: training history
    """
    X = df.to_numpy().reshape(-1, 935, 1)
    input_layer = Input(shape=(935, 1), name='input_data')
    x = layers.Conv1D(64, 12, activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, 12, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(935, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    shared = layers.Dense(8, activation=None)(x)
    existence_out = layers.Activation('softmax', name='existence')(shared)
    mixed_ratio_out = layers.Activation('softmax', name='mixed_ratio')(shared)
    peak = layers.Dense(128, activation='relu')(x)
    peak = layers.Dense(256, activation='relu')(peak)
    peak = layers.Dense(128, activation='relu')(peak)
    peak_area_out = layers.Dense(1, activation=None, name='peak_area')(peak)
    model = models.Model(inputs=input_layer, outputs=[existence_out, mixed_ratio_out, peak_area_out])
    model.compile(
        optimizer='adam',
        loss={
            'existence': 'categorical_crossentropy',
            'mixed_ratio': 'categorical_crossentropy',
            'peak_area': 'mse'
        },
        loss_weights={
            'existence': 1.0,
            'mixed_ratio': 1.0,
            'peak_area': 1.0
        },
        metrics={
            'existence': 'accuracy',
            'mixed_ratio': 'accuracy',
            'peak_area': 'mse'
        }
    )
    history = model.fit(
        X,
        {
            'existence': existence,
            'mixed_ratio': mixed_ratios,
            'peak_area': peak_areas
        },
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=2
    )
    return model, history
