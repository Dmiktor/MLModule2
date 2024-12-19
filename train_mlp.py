from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_mlp_model(input_shape, num_classes, learning_rate=0.001, dropout_rate=0.2):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Wrap the model with KerasClassifier, properly passing learning_rate
def create_mlp_classifier(X_train, num_classes, learning_rate=0.001):
    input_shape = X_train.shape[1]
    model = KerasClassifier(model=build_mlp_model, input_shape=input_shape, num_classes=num_classes, learning_rate=learning_rate, epochs=50, batch_size=32, verbose=1)
    return model