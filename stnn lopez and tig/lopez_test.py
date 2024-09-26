import pandas as pd
import numpy as np
import sklearn
import pickle

from sklearn.model_selection import train_test_split

import tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from ._lopez_protocol_header_fields_TDL import LopezTDLModality
from ._modal_wrapper import ModalWrapper
#from modals._lopez_protocol_header_fields_TDL import LopezTDLModality

import trustee
from trustee.report.trust import TrustReport

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape


def wrap_with_reshape(model, dim1, dim2):
    """
    Wraps a model such that it accepts a 1D vector input and reshapes it into a 2D matrix before passing to the model.

    Parameters:
    model (tf.keras.Model): The model to be wrapped.
    dim1 (int): The number of rows for reshaping.
    dim2 (int): The number of columns for reshaping.

    Returns:
    Model: A new model that accepts a 1D vector and reshapes it into a 2D matrix of shape (dim1, dim2).
    """
    # Get the original input shape
    input_length = dim1 * dim2

    # Define the new input as a 1D vector of size `dim1 * dim2`
    input_layer = Input(shape=(input_length,), name="1D_input")

    # Reshape the 1D vector into a 2D matrix of shape (dim1, dim2)
    reshaped_input = Reshape((dim1, dim2))(input_layer)

    # Pass the reshaped input to the original model
    model_output = model(reshaped_input)

    # Create the new model that wraps the original model
    wrapped_model = Model(inputs=input_layer, outputs=model_output)

    return wrapped_model


def MyFeature(feature_index):
    pooled_array_size = 1500

    # calculate x,y coordinates
    y = feature_index // pooled_array_size
    x = feature_index
    x = x % pooled_array_size

    return f'({x},{y})'


def wrap_model_with_argmax(model):
    input_shape = src.INPUT_SHAPE
    new_input = Input(shape=input_shape)
    model_output = model(new_input)
    argmax_output = ArgmaxLayer()(model_output)
    new_model = Model(inputs=new_input, outputs=argmax_output)
    return new_model

def preprocess_data(df):
    """
    Preprocess the DataFrame by converting the 'Data' column to a 3D NumPy array and encoding the labels.

    Args:
        df (pd.DataFrame): DataFrame with 'Data' column containing arrays of arrays and 'Label' column.

    Returns:
        X (np.ndarray): 3D NumPy array of features.
        y (np.ndarray): 1D NumPy array of encoded labels.
    """
    # Convert 'Data' column from list of lists to a NumPy array of float32
    X = np.array(df['Data'].apply(lambda x: np.array(eval(x), dtype=np.float32)).tolist())

    # Convert 'Label' column to numeric classes
    y = df['Label'].factorize()[0].astype(int)

    return X, y

def main():
    num_packets = 32

    # Load the dataset from CSV
    df = pd.read_csv('lopez_features.csv')

    #print(df)

    # Get features and labels
    X, y = preprocess_data(df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate the LopezTDLModality model
    lopez_model = LopezTDLModality(packet_count=num_packets)

    print("got the model")

    # Compile the model
    lopez_model.model.compile(optimizer=Adam(),
                              loss=SparseCategoricalCrossentropy(),
                              metrics=[SparseCategoricalAccuracy()])

    print("Compiled the model")

    print(y_train.dtype)
    print(y_train.ndim)

    # Train the model
    lopez_model.model.fit(X_train, y_train, epochs=20, batch_size=64)

    print("Trained the model")

    # Evaluate the model
    loss, accuracy = lopez_model.model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    # Make predictions on the test set
    predictions = lopez_model.model.predict(X_test)

    # Convert logits to class predictions
    predicted_classes = np.argmax(predictions, axis=1)

    print(f"Predicted classes: {predicted_classes}")
    print(f"True classes: {y_test}")

    # Calculate accuracy manually to confirm the model's reported accuracy
    manual_accuracy = np.mean(predicted_classes == y_test)
    print(f"Manual Accuracy: {manual_accuracy}")


    wrapped_model = wrap_with_reshape(lopez_model.model, num_packets, 3)

    # Reshape the array to (samples, dim1 * dim2)
    reshaped_train = np.reshape(X_train, (X_train.shape[0], -1))
    reshaped_test = np.reshape(X_test, (X_test.shape[0], -1))

    trust_report = TrustReport(
        wrapped_model,
        X_train=reshaped_train,
        X_test=reshaped_test,
        y_train=y_train,
        y_test=y_test,
        top_k=10,
        max_iter=1,
        trustee_num_iter=5,#25
        trustee_num_stability_iter=2,#10
        num_pruning_iter=1, #10
        trustee_sample_size=0.3,
        analyze_stability=True,
        analyze_branches=True,
        skip_retrain=True,
        #class_names=["Youtube", "GoogleDoc", "GoogleDrive"],
        # logger=logger,
        verbose=False
    )

    #TODO feature names

    print(trust_report)
    trust_report.plot('/content/drive/MyDrive/Colab Notebooks/edited_flowpic_replication/data/trustee')
    print("plot done")
    trust_report.save('/content/drive/MyDrive/Colab Notebooks/edited_flowpic_replication/data/trustee')
    print("save done")

    #TODO save single tree model

if __name__ == '__main__':
    main()