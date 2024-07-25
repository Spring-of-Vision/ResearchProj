import src
from src.models.ingesting import get_dataset

import os
import glob
import numpy as np
import sklearn

from trustee.utils import log

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model

import trustee
from trustee import ClassificationTrustee
from trustee.report.trust import TrustReport

from sklearn.metrics import classification_report

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

    logger = log.Logger("data/trustee/output.log")

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

    all_files = glob.glob(os.path.join(datapath, '*/*.npy'))
    print(f'{len(all_files)} feature files found.')

    test, test_steps = get_dataset(all_files, batch_size, dimensions_to_use, for_prediction=True)

    # Iterate over the dataset and extract just the labels
    #labels = extract_labels(test, test_steps) #test.map(lambda data, label: label)
    #data = extract_data(test, test_steps)

    # Assuming labels is a tf.data.Dataset object containing labels
    #print(labels)

    # Reshape the array to (3, 1500 * 1500 * 1)
    #reshaped_data = data.reshape(data.shape[0], -1)

    print(test)

    # Iterate over the dataset and extract just the data
    #test = test.map(lambda data, label: data)
    test = test.prefetch(tf.data.experimental.AUTOTUNE)

    predictions = wrapped_model.predict(test, steps=test_steps) #changed from model to wrapped_model

    print(predictions)

    # Convert to shape (13, 1) by finding the index of the maximum value along axis 1
    #class_predictions = np.argmax(predictions, axis=1).reshape(-1, 1)
    #print(class_predictions)

    print("labels:")
    labels = get_labels(all_files)
    print(labels)
    print("Labels Shape:", labels.shape)
    print(len(labels))

    # Convert to shape (13, 1) by finding the index of the maximum value along axis 1
    class_labels = np.argmax(labels, axis=1).reshape(-1, 1)

    print(class_labels)

    X_test = to_numpy(test)
    print(len(X_test))

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

    trust_report = TrustReport(
        wrapped_model,
        X_train=X_test,
        X_test=X_test,
        y_train=class_labels,
        y_test=class_labels,
        top_k=10,
        max_iter=0,
        trustee_num_iter=1, #10
        num_pruning_iter=1, #30
        trustee_sample_size= 1, #0.3,
        analyze_stability=True,
        analyze_branches=True,
        skip_retrain=True,
        class_names= ["GoogleDrive", "GoogleDocs", "Youtube"],
        logger=logger,
        verbose=False
    )

    #print(trust_report)
    trust_report.plot('data/trustee')
    trust_report.save('data/trustee')
    logger.log(trust_report)

    metric_results = model.compute_metrics(test, class_labels, predictions, sample_weight=None) #?
    print(metric_results)

'''
    # Trustee
    trustee = ClassificationTrustee(expert=model)
    trustee.fit(X_test, labels, num_iter=50, num_stability_iter=10, samples_size=0.3, verbose=True)
    dt, pruned_dt, agreement, reward = trustee.explain()
    dt_y_pred = dt.predict(X_test)
    #pruned_dt_y_pred = pruned_dt.predict(test)

    print("Model explanation global fidelity report:")
    print(classification_report(predictions, dt_y_pred))
    print("Model explanation score report:")
    print(classification_report(labels, dt_y_pred))

'''