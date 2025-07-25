import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input

def wipe():
    """
    Wipe the current TensorFlow session.
    """
    tf.keras.backend.clear_session()

def kim2025(training_df, test_df, training_exp_df, test_exp_df):

    input_layer = Input(shape=(len(training_df.index), 1), name='input_data')

    # First Conv1D block
    x = layers.Conv1D(64, 12, activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    # Second Conv1D block
    x = layers.Conv1D(32, 12, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    # MaxPooling
    x = layers.MaxPooling1D(pool_size=2)(x)
    # Flatten
    x = layers.Flatten()(x)
    # Dense layers with Dropout
    x = layers.Dense(len(training_df.index), activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)

    # Shared output for Existence and Mixed ratio
    shared = layers.Dense(8, activation=None)(x)

    carbon = layers.Dense(128, activation='relu')(x)
    carbon = layers.Dense(256, activation='relu')(carbon)
    carbon = layers.Dense(128, activation='relu')(carbon)
    carbon = layers.Dense(1, activation=None, name='carbon')(carbon)
    # Existence head
    # existence_out = layers.Activation('softmax', name='existence')(shared)

    # Mixed ratio head
    # mixed_ratio_out = layers.Activation('softmax', name='mixed_ratio')(shared)

    # Peak area head
    # peak = layers.Dense(128, activation='relu')(x)
    # peak = layers.Dense(256, activation='relu')(peak)
    # peak = layers.Dense(128, activation='relu')(peak)
    # peak_area_out = layers.Dense(1, activation=None, name='peak_area')(peak)

    # Model
    model = models.Model(inputs=input_layer, outputs={
        'carbon': carbon,
        # 'existence': existence_out, 
        # 'mixed_ratio': mixed_ratio_out, 
        # 'peak_area': peak_area_out
    })

    x_train = training_df.to_numpy().reshape(-1, len(training_df.index), 1).astype('float32')
    y_train_carbon = training_exp_df['avg_carbon_portion'].values.astype('float32')

    x_test = test_df.to_numpy().reshape(-1, len(test_df.index), 1).astype('float32')
    y_test_carbon = test_exp_df['avg_carbon_portion'].values.astype('float32')
    # print("Training data shape:", x_train.shape)
    # print("Training data x first value:", x_train[0][0][0])
    # print("Training carbon values shape:", y_train_carbon.shape)
    # print("Training carbon first value:", y_train_carbon[0])

    # return x_train, y_train_carbon
    # y_existence = existence  # this is the NumPy array from cell 2
    # y_mixed_ratio = mixed_ratios  # shape (num_samples, 8)
    # y_peak_area = peak_areas  # shape (num_samples, 1)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss={
            'carbon': 'mse',
            # 'existence': 'categorical_crossentropy',
            # 'mixed_ratio': 'categorical_crossentropy',
            # 'peak_area': 'mse'
        },
        loss_weights={
            'carbon': 1.0,
            # 'existence': 1.0,
            # 'mixed_ratio': 1.0,
            # 'peak_area': 1.0
        },
        metrics={
            'carbon': 'mse',
            # 'existence': 'accuracy',
            # 'mixed_ratio': 'accuracy',
            # 'peak_area': 'mse'
        }
    )
    history = model.fit(
        x_train,
        {
            'carbon': y_train_carbon,
            # 'existence': y_existence,
            # 'mixed_ratio': y_mixed_ratio,
            # 'peak_area': y_peak_area
        },
        epochs=100,
        batch_size=16,
        validation_split=0.2
    )

    combined_df = pd.concat([training_df, test_df], axis=1)
    # unique columns
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    combined_X = combined_df.to_numpy().reshape(-1, len(combined_df.index), 1).astype('float32')
    predictions = model.predict(combined_X)
    # predictions = model.predict(x_test)

    carbon_predictions = predictions['carbon'].flatten()
    carbon_predictions_df = pd.DataFrame({
        'Carbon Portion': carbon_predictions
    }, index=combined_df.columns)

    wipe()
    return carbon_predictions_df, history