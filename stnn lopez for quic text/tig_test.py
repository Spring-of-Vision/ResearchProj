import pandas as pd
import numpy as np
from ._graphdapp import GraphDAppModality
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


def main():
    # Load the dataset from CSV
    df = pd.read_csv('tig_features.csv')

    nodes = 32

    # Assuming df is your dataframe with columns: 'label', 'adj_matrix', 'node_features'
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Extracting input features and labels
    X_train_adj = np.array(train_df['Adj_Matrix'].apply(lambda x: np.array(eval(x), dtype=np.int8)).tolist())  # Adjacency matrices
    X_train_features = np.array(train_df['Features'].apply(lambda x: np.array(eval(x), dtype=np.int32)).tolist())  # Node features
    y_train = pd.factorize(train_df['Label'])[0]

    #y_train = to_categorical(y_train)

    X_test_adj = np.array(test_df['Adj_Matrix'].apply(lambda x: np.array(eval(x), dtype=np.int8)).tolist())
    X_test_features = np.array(test_df['Features'].apply(lambda x: np.array(eval(x), dtype=np.int32)).tolist())
    y_test = pd.factorize(test_df['Label'])[0]

    #y_test = to_categorical(y_test)

    print(y_train.dtype)
    print(y_train.ndim)
    print(y_train)

    # Initialize the model
    graphdapp_model  = GraphDAppModality(n_packets=nodes)

    # Compile the model
    graphdapp_model.model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fitting the model
    graphdapp_model.model.fit(
        [X_train_adj, X_train_features],  # List of inputs
        y_train,  # Labels
        validation_data=([X_test_adj, X_test_features], y_test),
        epochs=20,  # Number of epochs
        batch_size=32  # Batch size
    )

    # Train the model
    #graphdapp_model.model.fit(X_train, y_train, epochs=20, batch_size=64) #, validation_split=0.2)

    # Predicting on the test data
    y_pred = graphdapp_model.model.predict([X_test_adj, X_test_features])

    # Convert predictions from one-hot encoding to class indices
    y_pred = np.argmax(y_pred, axis=1)
    #y_test = np.argmax(y_test, axis=1)

    print(y_pred.dtype)
    print(y_test.dtype)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.4f}')

    print(f"Predicted classes: {y_pred}")
    print(f"True classes: {y_test}")

    # Calculate accuracy manually to confirm the model's reported accuracy
    manual_accuracy = np.mean(y_pred == y_test)
    print(f"Manual Accuracy: {manual_accuracy}")

if __name__ == '__main__':
    main()