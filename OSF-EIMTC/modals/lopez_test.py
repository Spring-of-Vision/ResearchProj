import pandas as pd
import numpy as np
import sklearn
import pickle
import os
import shutil
import pydotplus

from sklearn.model_selection import train_test_split
from sklearn import tree

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from ._lopez_protocol_header_fields_TDL import LopezTDLModality
from ._modal_wrapper import ModalWrapper
#from modals._lopez_protocol_header_fields_TDL import LopezTDLModality

import trustee
from trustee.utils import log
from trustee.report.trust import TrustReport
from trustee.utils.tree import get_dt_info, top_k_prune
import graphviz

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Reshape

def MyFeature(feature_index):

    # calculate x,y coordinates
    rows = feature_index // 3
    cols = feature_index % 3

    if cols == 0:
        stat = 'time'
    elif cols == 1:
        stat = 'size'
    else:
        stat = 'direction'

    return f'packet {rows}, {stat}'

# Custom Argmax Layer
class ArgmaxLayer(Layer):
    def __init__(self, **kwargs):
        super(ArgmaxLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.argmax(inputs, axis=1, output_type=tf.int32)


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

    # Argmax using custom layer
    argmax_output = ArgmaxLayer()(model_output)

    # Create the new model that wraps the original model
    new_model = Model(inputs=input_layer, outputs=argmax_output)

    return new_model


def get_stats(trust_report):
    all_trees = trust_report.trustee.get_all_students()
    num_stability_iter = trust_report.trustee_num_stability_iter
    features = trust_report.feature_names
    classes = trust_report.class_names

    path = "./trustee-lopez/student_trees"

    # Check if the directory exists and delete the directory and its contents
    if os.path.exists(path):
        shutil.rmtree(path)

    # Create the directory again
    os.makedirs(path, exist_ok=True)

    all_features = {}
    i = 0
    for j in range(num_stability_iter):
        for dt, rev in all_trees[j]:
            dot_data = tree.export_graphviz(
                dt,
                class_names=classes,
                feature_names=features,
                filled=True,
                rounded=True,
                special_characters=True)
            graph = graphviz.Source(dot_data)
            graph.render(f'{path}/student_tree_{i}')

            features_used, splits, branches = get_dt_info(dt)

            for feat in features_used:
                if feat not in all_features:
                    all_features[feat] = {"feat_name": MyFeature(feat), "count_total": 0, "num_trees": 0, "samples": 0}

                all_features[feat]["count_total"] += features_used[feat]["count"]
                all_features[feat]["num_trees"] += 1
                all_features[feat]["samples"] += features_used[feat]["samples"]

            i += 1

    # save in txt file
    with open('./trustee-lopez/stats.txt', 'w') as file:
        file.write("all tree:\n")
        for feat in all_features:
            file.write(
                f'feat_name: {all_features[feat]["feat_name"]}, count_total: {all_features[feat]["count_total"]}, num_trees: {all_features[feat]["num_trees"]}, samples: {all_features[feat]["samples"]}\n')

        file.write("\n")

        sorted_features = sorted(all_features.items(), key=lambda x: (x[1]["num_trees"], x[1]["samples"]), reverse=True)
        file.write("sorted features:\n")
        for feat, data in sorted_features:
            file.write(
                f'feat_name: {data["feat_name"]}, count_total: {data["count_total"]}, num_trees: {data["num_trees"]}, samples: {data["samples"]}\n')

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
    #y = df['Label'].factorize()[0].astype(int)
    y = pd.factorize(df['Label'])[0].astype(int)

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

    # Train the model
    lopez_model.model.fit(X_train, y_train, epochs=20, batch_size=64)

    print("Trained the model")

    # Evaluate the model
    loss, accuracy = lopez_model.model.evaluate(X_test, y_test)

    # Make predictions on the test set
    predictions = lopez_model.model.predict(X_test)

    # Convert logits to class predictions
    predicted_classes = np.argmax(predictions, axis=1)

    print(f"Predicted classes: {predicted_classes}")
    print(f"True classes: {y_test}")

    # Calculate accuracy manually to confirm the model's reported accuracy
    manual_accuracy = np.mean(predicted_classes == y_test)

    wrapped_model = wrap_with_reshape(lopez_model.model, num_packets, 3)

    # Reshape the array to (samples, dim1 * dim2)
    reshaped_train = np.reshape(X_train, (X_train.shape[0], -1))
    reshaped_test = np.reshape(X_test, (X_test.shape[0], -1))

    print(type(y_train))
    print(y_train.dtype)
    print(y_train.ndim)
    print(y_train)
    print(type(y_test))
    print(y_test.dtype)
    print(y_test.ndim)
    print(y_test)

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
        class_names=["Youtube", "GoogleDoc", "GoogleDrive"],
        # logger=logger,
        verbose=False
    )

    #print(X_train.shape)

    # feature names
    if trust_report.feature_names is None:
        trust_report.feature_names = [MyFeature(i) for i in range(96)]

    print("===============  here   ===================")

    #print(trust_report.feature_names)

    trust_report.plot('./trustee-lopez')
    print("plot done")
    trust_report.save('./trustee-lopez')
    print("save done")

    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    print(f"Manual Accuracy: {manual_accuracy}")

    print(trust_report)

    # Save stats and single tree
    get_stats(trust_report)

    dt, purnd_dt, agreement, reward = trust_report.trustee.explain()
    # Save the dt model to a file
    with open(
            f"./trustee-lopez/decision_tree_model.pkl",
            'wb') as file:
        pickle.dump(dt, file)

if __name__ == '__main__':
    main()