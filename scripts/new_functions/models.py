import tensorflow as tf
from tensorflow.keras import layers, models


def build_fully_connected_model(
    input_dim: int,
    hidden_layer_dims: tuple[int, ...],
    output_dim: int,
) -> tf.keras.Model:
    """
    Builds and compiles a fully connected neural network Keras model.

    :param input_dim: Input is a 1d array of length input_dim.
    :param hidden_layer_dims: A tuple specifying the number of nodes in each hidden layer
    :param output_dim: Dimension of the output layer

    :return: A compiled TensorFlow Keras Model ready for training
    """

    # Define the input layer
    keras_input = tf.keras.Input(
        shape=(input_dim,)
    )  # Somehow breaks if shape=(input_dim, 1) is speficied instead

    # Build the hidden layers sequentially as specified by hidden_layer_dims
    x = keras_input
    for dim in hidden_layer_dims:
        x = layers.Dense(dim, activation="relu")(x)
    hidden_layer_result = x

    # Define the output layer with linear activation (None euals linear activation)
    output_layer = layers.Dense(output_dim, activation=None)(hidden_layer_result)

    model = tf.keras.Model(inputs=keras_input, outputs=output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(amsgrad=True),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model


def build_convolution_model(
    input_dim: int,
    hidden_layer_filters: tuple[int, ...],
    output_dim: int,
    kernel_size: int = 3,
    activation_function: str = "relu",
) -> tf.keras.Model:
    """
    Builds and compiles a 1D convolutional model for regression.

    :param input_dim: Input is a 1d array of length input_dim.
    :param output_dim: The dimension of the output layer.
    :param kernel_size: The size of the convolution kernel.
    :param activation_function: The activation function to use in the Conv1D layers.

    :return: A compiled 1D convolutional tf.keras.Model neural network model
    """

    # Define the input layer
    keras_input = tf.keras.Input(
        shape=(input_dim, 1)
    )  # Somehow breaks if shape=(input_dim,) is speficied instead
    x = keras_input

    # Build the hidden layers sequentially as specified by hidden_layer_filters
    for n_filters in hidden_layer_filters:
        x = layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding="same",
            activation=activation_function,
        )(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

    # Flatten and final dense layer with linear activation (None euals linear activation)
    x = layers.Flatten()(x)
    outputs = layers.Dense(output_dim, activation=None)(x)

    model = models.Model(inputs=keras_input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(amsgrad=True),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model
