from ._modal_wrapper import ModalWrapper
from ._stnn import STNNModality
import pandas as pd
import numpy as np
import os
import shutil
import pickle

from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Reshape

from trustee.report.trust import TrustReport
from trustee.utils.tree import get_dt_info, top_k_prune
import graphviz

def MyFeature(feature_index):

    # calculate x,y coordinates
    rows = feature_index // 3
    cols = feature_index % 3

    if rows == 0:
        stat = 'bi'
    elif rows == 1:
        stat = '0' # direction 0- server to client
    else:
        stat = '1' # direction 1- client to server

    # The 14 stats
    strings = [
        "mean time", "max time", "min time", "time stddev",
        "time skew", "mean size", "max size", "min size",
        "size stddev", "size skew", "num packets", "total bytes",
        "packets/sec", "bytes/sec"
    ]

    return f'dir: {stat}, {strings[cols]}'

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

    path = "./trustee-stnn/student_trees"

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
    with open('./trustee-stnn/stats.txt', 'w') as file:
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
    Preprocesses the data for training and testing.
    """
    # Convert 'Data' column from list of lists to a NumPy array of float32
    X = np.array(df['Matrix'].apply(lambda x: np.array(eval(x), dtype=np.float32)).tolist())
    y = pd.factorize(df['Label'])[0]  # Convert labels to integers

    # Convert labels to one-hot encoding
    y = to_categorical(y)
    y = np.argmax(y, axis=1)

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
    stnn_model.model.compile(optimizer=Adam(learning_rate=0.00007), loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])

    # Compile the model
    #stnn_model.model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=[SparseCategoricalAccuracy()])

    # Train the model
    stnn_model.model.fit(X_train, y_train, epochs=20, batch_size=64) #, validation_split=0.2)

    #y_test_one_hot = y_test

    # Predict using the test set
    y_pred = stnn_model.model.predict(X_test)

    # Convert predictions from one-hot encoding to class indices
    y_pred = np.argmax(y_pred, axis=1)
    #y_test = np.argmax(y_test, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.4f}')

    print(f"Predicted classes: {y_pred}")
    print(f"True classes: {y_test}")

    # Calculate accuracy manually to confirm the model's reported accuracy
    manual_accuracy = np.mean(y_pred == y_test)
    print(f"Manual Accuracy: {manual_accuracy}")

    wrapped_model = wrap_with_reshape(stnn_model.model, 3, 14)

    # Reshape the array to (samples, dim1 * dim2)
    reshaped_X_train = np.reshape(X_train, (X_train.shape[0], -1))
    reshaped_X_test = np.reshape(X_test, (X_test.shape[0], -1))

    print(reshaped_X_train.dtype)
    print(len(reshaped_X_train))
    print(len(y_train))
    print(len(reshaped_X_test))
    print(len(y_test))

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
        X_train=reshaped_X_train,
        X_test=reshaped_X_test,
        y_train=y_train,
        y_test=y_test,
        top_k=10,
        max_iter=1,
        trustee_num_iter=5,  # 25
        trustee_num_stability_iter=2,  # 10
        num_pruning_iter=1,  # 10
        trustee_sample_size=0.3,
        analyze_stability=True,
        analyze_branches=True,
        skip_retrain=True,
        class_names=["Youtube", "GoogleDoc", "GoogleDrive"],
        # logger=logger,
        verbose=False
    )

    # feature names
    #if trust_report.feature_names is None:
    #    trust_report.feature_names = [f'{MyFeature(i)}' for i in range(42)]

    trust_report.plot('./trustee-stnn')
    print("plot done")
    trust_report.save('./trustee-stnn')
    print("save done")

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

if __name__ == "__main__":
    main()