import pandas as pd
import numpy as np
import pickle
import os
import shutil

from ._graphdapp import GraphDAppModality
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Reshape

import trustee
from trustee.utils import log
from trustee.report.trust import TrustReport
from trustee.utils.tree import get_dt_info, top_k_prune
import graphviz

def MyFeature(feature_index, nodes):

# nodes * nodes + nodes

    if feature_index in range(nodes*nodes):
        # calculate x,y coordinates
        rows = feature_index // 32
        cols = feature_index % 32
        return f'edge_{rows}X{cols}'
    else:
        feature = feature_index - (nodes*nodes)
        return f'pkt_{feature}' # will display packet feature as size*dir

# Custom Argmax Layer
class ArgmaxLayer(Layer):
    def __init__(self, **kwargs):
        super(ArgmaxLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.argmax(inputs, axis=1, output_type=tf.int32)


def wrap_with_reshape(model, nodes):
    """
    Wraps a model such that it accepts a 1D vector input and reshapes it into a 2D matrix before passing to the model.

    Parameters:
    model (tf.keras.Model): The model to be wrapped.
    dim1 (int): The number of rows for reshaping.
    dim2 (int): The number of columns for reshaping.

    Returns:
    Model: A new model that accepts a 1D vector and reshapes it into a 2D matrix of shape (dim1, dim2).
    """

    input_shape = ((nodes*nodes+nodes),)
    new_input = Input(shape=input_shape)

    # Pass the input to the original model
    model_output = model(new_input)

    # Argmax using custom layer
    argmax_output = ArgmaxLayer()(model_output)

    # Create the new model that wraps the original model
    new_model = Model(inputs=new_input, outputs=argmax_output)

    return new_model

def get_stats(trust_report):
    all_trees = trust_report.trustee.get_all_students()
    num_stability_iter = trust_report.trustee_num_stability_iter
    features = trust_report.feature_names
    classes = trust_report.class_names

    path = "./trustee-tig/student_trees"

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
    with open('./trustee-tig/stats.txt', 'w') as file:
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


def main():
    # Load the dataset from CSV
    df = pd.read_csv('tig_features.csv')

    nodes = 32

    # Assuming df is your dataframe with columns: 'label', 'adj_matrix', 'node_features'
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Extracting input features and labels
    X_train_adj = np.array(train_df['Adj_Matrix'].apply(lambda x: np.array(eval(x), dtype=np.int8)).tolist())  # Adjacency matrices
    #X_train_features = np.array(train_df['Features'].apply(lambda x: np.array(eval(x), dtype=np.int32)).tolist())  # Node features
    X_train_features = np.array(
        train_df['Features'].apply(lambda x: np.array(eval(x), dtype=np.int32).reshape(-1)).tolist())

    y_train = pd.factorize(train_df['Label'])[0]

    #y_train = to_categorical(y_train)

    X_test_adj = np.array(test_df['Adj_Matrix'].apply(lambda x: np.array(eval(x), dtype=np.int8)).tolist())
    #X_test_features = np.array(test_df['Features'].apply(lambda x: np.array(eval(x), dtype=np.int32)).tolist())
    X_test_features = np.array(
        test_df['Features'].apply(lambda x: np.array(eval(x), dtype=np.int32).reshape(-1)).tolist())
    y_test = pd.factorize(test_df['Label'])[0]

    #y_test = to_categorical(y_test)

    # Reshape the array to (samples, dim1 * dim2)
    reshaped_train_adj = np.reshape(X_train_adj, (X_train_adj.shape[0], -1))
    reshaped_test_adj = np.reshape(X_test_adj, (X_test_adj.shape[0], -1))

    print(reshaped_train_adj.shape)
    print(X_train_features.shape)

    new_X_train = np.concatenate((reshaped_train_adj,X_train_features), axis=1)
    new_X_test = np.concatenate((reshaped_test_adj,X_test_features), axis=1)

    # Initialize the model
    graphdapp_model = GraphDAppModality(n_packets=nodes)

    # Compile the model
    graphdapp_model.model.compile(optimizer=Adam(learning_rate=0.0001), loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])



    # Fitting the model
    graphdapp_model.model.fit(
        new_X_train,  # List of inputs
        y_train,  # Labels
        epochs=20,  # Number of epochs
        batch_size=32  # Batch size
    )

    # Train the model
    #graphdapp_model.model.fit(X_train, y_train, epochs=20, batch_size=64) #, validation_split=0.2)

    # Predicting on the test data
    y_pred = graphdapp_model.model.predict(new_X_test)

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

    #linear_X_train = np.concatenate(reshaped_train_adj, X_train_features)
    #linear_X_test = np.concatenate(reshaped_test_adj, X_test_features)

    wrapped_model = wrap_with_reshape(graphdapp_model.model, nodes)

    trust_report = TrustReport(
        wrapped_model,
        X_train=new_X_train,
        X_test=new_X_test,
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
    #    trust_report.feature_names = [MyFeature(i, nodes) for i in range(nodes*nodes+nodes)]

    trust_report.plot('./trustee-tig')
    print("plot done")
    trust_report.save('./trustee-tig')
    print("save done")

    #rint(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    print(f"Manual Accuracy: {manual_accuracy}")

    print(trust_report)

    # Save stats and single tree
    get_stats(trust_report)

    dt, purnd_dt, agreement, reward = trust_report.trustee.explain()
    # Save the dt model to a file
    with open(
            f"./trustee-tig/decision_tree_model.pkl",
            'wb') as file:
        pickle.dump(dt, file)

if __name__ == '__main__':
    main()