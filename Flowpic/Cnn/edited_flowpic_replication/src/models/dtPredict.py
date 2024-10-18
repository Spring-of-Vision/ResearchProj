import warnings

warnings.filterwarnings("ignore")

import src
from src.models.ingesting import get_dataset

import os
import shutil
import glob
import numpy as np
import sklearn
import pydotplus
import graphviz
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model

from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree

import json


# Load test_files from a JSON file
def load_files(filepath):
    with open(filepath, 'r') as f:
        test_files = json.load(f)
    return test_files

def to_numpy(data):
    data_list = []

    for element in data:
        data_list.append(element[0].numpy())

    # Convert lists to numpy arrays
    data_array = np.array(data_list)

    print("Data Array Shape:", data_array.shape)

    return data_array

 
    
def get_labels(file_list):
    labels = []
    for file in file_list:
        # One-hot encoding for labels
        label = [0, 0, 0]
        if os.path.dirname(file).endswith('GoogleDrive'):
            label[0] = 1
        elif os.path.dirname(file).endswith('GoogleDoc'):
            label[1] = 1
        elif os.path.dirname(file).endswith('Youtube'):
            label[2] = 1
        labels.append(label)

    labels = np.asarray(labels)
    return labels


def dtpred():

    #logger = log.Logger("/content/drive/MyDrive/Colab Notebooks/edited_flowpic_replication/src/data/trustee/output.log")

    # Load the trained model
    with open(f"/content/drive/MyDrive/Colab Notebooks/edited_flowpic_replication/data/trustee/decision_tree_model.pkl", "rb") as file:
      dt_model = pickle.load(file)


    test_files = load_files('/content/drive/MyDrive/Colab Notebooks/edited_flowpic_replication/config/test_files.json')
    print(f'{len(test_files)} feature files found.')
    test, test_steps = get_dataset(test_files, 1, 1, for_prediction=True)

    print(test)

    print("test_labels:")
    test_labels = get_labels(test_files)
    print(test_labels)
    print("Labels Shape:", test_labels.shape)
    print(len(test_labels))

    # Convert to shape (13, 1) by finding the index of the maximum value along axis 1
    y_test = np.argmax(test_labels, axis=1).reshape(-1, 1)

    print(y_test)


    # Iterate over the dataset and extract just the data
    #test = test.map(lambda data, label: data)
    test = test.prefetch(tf.data.experimental.AUTOTUNE)

    X_test = to_numpy(test)
    print(len(X_test))

    print(f'X_test.shape: {X_test.shape[1]}')

    y_pred = dt_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy:.4f}')
