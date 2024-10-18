import warnings

warnings.filterwarnings("ignore")

import src
from src.models.ingesting import get_dataset
from sklearn.metrics import accuracy_score

import os
import shutil
import glob
import numpy as np
import sklearn
import pydotplus
import graphviz
import pickle

from trustee.utils import log
from trustee.utils.tree import get_dt_info, top_k_prune

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model

import trustee
from trustee import ClassificationTrustee
from trustee.report.trust import TrustReport

from sklearn.metrics import classification_report
from sklearn import tree

import json


# Load test_files from a JSON file
def load_files(filepath):
    with open(filepath, 'r') as f:
        test_files = json.load(f)
    return test_files

'''
def extract_labels(dataset, num_batches):
    labels = []
    #print("BEFORE FOR")
    #i = 0
    for _, label_batch in dataset.take(num_batches):
        #i += 1
        #print(i)
        #print("IN FOR")
        #print(label_batch)
        labels.extend(label_batch.numpy())
    return np.array(labels)

def extract_data(dataset, num_batches):
    data = []
    print("BEFORE FOR")
    i = 0
    for data_batch, _ in dataset.take(num_batches):
        i += 1
        print(i)
        print("IN FOR")
        #print(data_batch)
        data.extend(data_batch.numpy())
    return np.array(data)
'''

def to_numpy(data):
    data_list = []

    for element in data:
        data_list.append(element[0].numpy())

    # Convert lists to numpy arrays
    data_array = np.array(data_list)

    print("Data Array Shape:", data_array.shape)

    return data_array

def MyFeature(feature_index):
    
    pooled_array_size = 32 
    
    # calculate x,y coordinates
    y = feature_index // pooled_array_size
    x = feature_index
    x = x % pooled_array_size
    
    return f'({x},{y})'

def get_stats(trust_report):
    
    all_trees = trust_report.trustee.get_all_students()  
    num_stability_iter= trust_report.trustee_num_stability_iter
    features= trust_report.feature_names
    classes= trust_report.class_names

    path = "/content/drive/MyDrive/Colab Notebooks/edited_flowpic_replication/data/trustee/student_trees"
    
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
    with open('/content/drive/MyDrive/Colab Notebooks/edited_flowpic_replication/data/trustee/stats.txt', 'w') as file:
        file.write("all tree:\n")    
        for feat in all_features:   
            file.write(f'feat_name: {all_features[feat]["feat_name"]}, count_total: {all_features[feat]["count_total"]}, num_trees: {all_features[feat]["num_trees"]}, samples: {all_features[feat]["samples"]}\n')
        
        file.write("\n")
        
        sorted_features = sorted(all_features.items(), key=lambda x: (x[1]["num_trees"], x[1]["samples"]), reverse=True)
        file.write("sorted features:\n")
        for feat, data in sorted_features:
            file.write(f'feat_name: {data["feat_name"]}, count_total: {data["count_total"]}, num_trees: {data["num_trees"]}, samples: {data["samples"]}\n')

    
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

# Custom Argmax Layer
class ArgmaxLayer(Layer):
    def __init__(self, **kwargs):
        super(ArgmaxLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.argmax(inputs, axis=1, output_type=tf.int32)

def wrap_model_with_argmax(model):
    input_shape = src.INPUT_SHAPE
    new_input = Input(shape=input_shape)
    model_output = model(new_input)
    argmax_output = ArgmaxLayer()(model_output)
    new_model = Model(inputs=new_input, outputs=argmax_output)
    return new_model


def pred(modelpath, weightspath, datapath, batch_size, dimensions_to_use):

    #logger = log.Logger("/content/drive/MyDrive/Colab Notebooks/edited_flowpic_replication/src/data/trustee/output.log")

    # Load the trained model
    model = load_model(modelpath)
    model.load_weights(weightspath)

    '''
    # Compile the model (same as in training- unneccessary really)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.categorical_crossentropy, #changed from binary_crossentropy
        metrics=['accuracy']
    )
    '''

    wrapped_model = wrap_model_with_argmax(model)

    '''
    # Compile the wrapped model (optional, as no training is involved)
    wrapped_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # Use sparse_categorical_crossentropy since the output is class indices
        metrics=['accuracy']
    )
    '''

    #all_files = glob.glob(os.path.join(datapath, '*/*.npy'))
    #print(f'{len(all_files)} feature files found.')

    train_files = load_files('/content/drive/MyDrive/Colab Notebooks/edited_flowpic_replication/config/train_files.json')
    train_labels = get_labels(train_files)
    # Convert to shape (13, 1) by finding the index of the maximum value along axis 1
    y_train = np.argmax(train_labels, axis=1).reshape(-1, 1)
    print(f'y_train len: {len(y_train)}')

    train, train_steps = get_dataset(train_files, batch_size, dimensions_to_use, for_prediction=True)
    train = train.prefetch(tf.data.experimental.AUTOTUNE)
    X_train = to_numpy(train)
    print(f'X_train len: {len(X_train)}')

    test_files = load_files('/content/drive/MyDrive/Colab Notebooks/edited_flowpic_replication/config/test_files.json')
    print(f'{len(test_files)} feature files found.')
    test, test_steps = get_dataset(test_files, batch_size, dimensions_to_use, for_prediction=True)

    # Iterate over the dataset and extract just the labels
    #labels = extract_labels(test, test_steps) #test.map(lambda data, label: label)
    #data = extract_data(test, test_steps)

    # Assuming labels is a tf.data.Dataset object containing labels
    #print(labels)

    # Reshape the array to (3, 1500 * 1500 * 1)
    #reshaped_data = data.reshape(data.shape[0], -1)

    print(test)

    # Convert to shape (13, 1) by finding the index of the maximum value along axis 1
    #class_predictions = np.argmax(predictions, axis=1).reshape(-1, 1)
    #print(class_predictions)

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

    predictions = wrapped_model.predict(test, steps=test_steps) #changed from model to wrapped_model

    print(predictions)
    X_test = to_numpy(test)
    print(f"X_test len: {len(X_test)}")

    print(f'X_train.shape: {X_train.shape[1]}')
    print(f'X_test.shape: {X_test.shape[1]}')
    '''
    # Reshape the array
    num_examples = X_test.shape[0]
    rows = X_test.shape[1]
    cols = X_test.shape[2]
    reshaped_data = X_test.reshape(num_examples, rows * cols)
    '''

    #reshaped_data = reshaped_data / reshaped_data.max()
    #reshaped_data = reshaped_data.astype(np.float32)
    #reshaped_data = reshaped_data.multiply(reshaped_data, 1.0 / 255.0)

    print(type(y_train))
    print(y_train.dtype)
    print(y_train.ndim)
    print(y_train[0])
    print(type(y_test))
    print(y_test.dtype)
    print(y_test.ndim)
    print(y_test[0])

    trust_report = TrustReport(
        wrapped_model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        top_k=10,
        max_iter=1,
        trustee_num_iter=25,
        trustee_num_stability_iter=10,
        num_pruning_iter=10,
        trustee_sample_size=0.3,
        analyze_stability=True,
        analyze_branches=True,
        skip_retrain=True,
        class_names= ["GoogleDrive", "GoogleDoc", "Youtube"],
        #logger=logger,
        verbose=False
    )

    # אפשר גם שנגדיר ישר בסטראקט פה
    if trust_report.feature_names is None:
      trust_report.feature_names = [f"feature_{MyFeature(i)}" for i in range(X_train.shape[1])]
      
    print(trust_report)
    trust_report.plot('/content/drive/MyDrive/Colab Notebooks/edited_flowpic_replication/data/trustee')
    print("plot done")

    trust_report.save('/content/drive/MyDrive/Colab Notebooks/edited_flowpic_replication/data/trustee')
    #trust_report._save_dts('/content/drive/MyDrive/Colab Notebooks/edited_flowpic_replication/data/trustee')
    print("save done")
    #logger.log(trust_report)

    dt, purnd_dt, agreement, reward =trust_report.trustee.explain()
    # Save the dt model to a file
    with open(f"/content/drive/MyDrive/Colab Notebooks/edited_flowpic_replication/data/trustee/decision_tree_model.pkl", 'wb') as file:
      pickle.dump(dt, file)

    get_stats(trust_report)

    print(trust_report)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    print(f'Accuracy: {accuracy:.4f}')

    """  only for the metric_results
    y_test = tf.convert_to_tensor(y_test)
    predictions = tf.convert_to_tensor(predictions)    
    
    print(f'X_test: {type(X_test)}')
    print(f'X_test: {X_test}')
    print(f'X_test: {X_test.shape}')
    print(f'y_test: {type(y_test)}')
    print(f'y_test: {y_test}')
    print(f'y_test: {y_test.shape}')
    print(f'predictions :{type(predictions)}')
    print(f'predictions :{predictions}')
    print(f'predictions :{predictions.shape}')

    #print(f'test_labels shape: {test_labels.shape}')
    metric_results = model.compute_metrics(X_test, y_test, predictions, sample_weight=None)
    print(metric_results)
    """