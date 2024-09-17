from ._modal_wrapper import ModalWrapper
from ._stnn import STNNModality
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

def preprocess_data(df):
    """
    Preprocesses the data for training and testing.
    """
    # Convert 'Data' column from list of lists to a NumPy array of float32
    X = np.array(df['Matrix'].apply(lambda x: np.array(eval(x), dtype=np.float32)).tolist())
    y = pd.factorize(df['Label'])[0]  # Convert labels to integers

    # Convert labels to one-hot encoding
    y = to_categorical(y)

    # Ensure X shape is correct
    print(f'Shape of X: {X.shape}')  # Should be (num_samples, 3, 14)
    print(f'Shape of y: {y.shape}')  # Should be (num_samples, num_classes)or (num_samples)
    return X, y

def main():
    # Load the dataset from CSV
    df = pd.read_csv('stnn_features.csv')

    # Preprocess the data
    X, y = preprocess_data(df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    stnn_model = STNNModality()

    # Compile the model
    stnn_model.model.compile(optimizer=Adam(learning_rate=0.00007), loss='categorical_crossentropy', metrics=['accuracy'])

    # Compile the model
    #stnn_model.model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=[SparseCategoricalAccuracy()])

    # Train the model
    stnn_model.model.fit(X_train, y_train, epochs=20, batch_size=64) #, validation_split=0.2)

    # Predict using the test set
    y_pred = stnn_model.model.predict(X_test)

    # Convert predictions from one-hot encoding to class indices
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.4f}')

    print(f"Predicted classes: {y_pred}")
    print(f"True classes: {y_test}")

    # Calculate accuracy manually to confirm the model's reported accuracy
    manual_accuracy = np.mean(y_pred == y_test)
    print(f"Manual Accuracy: {manual_accuracy}")


if __name__ == "__main__":
    main()