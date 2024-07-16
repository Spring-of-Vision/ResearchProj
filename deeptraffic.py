# Code below was copied and adapted from the original code for DeepTraffic was provided by
#
# Wei Wang (ww8137@mail.ustc.edu.cn)
#
# Available at https://github.com/echowei/DeepTraffic
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file, You
# can obtain one at http://mozilla.org/MPL/2.0/.
# ==============================================================================

import os
import numpy as np

import input_data

# start tensorflow interactiveSession
import tensorflow.compat.v1 as tf  # load MNIST data


tf.compat.v1.disable_eager_execution()

CLASS_NUM = 3
TRAIN_ROUND = 100  #20000


# Xavier initialization
def xavier_init(shape):
    in_dim = shape[0]
    out_dim = shape[1]
    stddev = tf.sqrt(2.0 / (in_dim + out_dim))
    initial = tf.random_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return -1


# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding="SAME")


class DeepTraffic:
    def __init__(self):
        self.sess = tf.InteractiveSession()

        # Create the model
        # placeholder
        self.x = tf.placeholder("float", [None, 1500*1500])
        self.y_ = tf.placeholder("float", [None, CLASS_NUM])

        # first convolutinal layer
        self.w_conv1 = xavier_init([1, 25, 1, 32])
        self.b_conv1 = bias_variable([32])

        self.x_image = tf.reshape(self.x, [-1, 1, 1500*1500, 1])
        print(f'x_image: {self.x_image.get_shape()}')

        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.w_conv1) + self.b_conv1)
        print(f'h_conv1: {self.h_conv1.get_shape()}')
        self.h_pool1 = max_pool_2x2(self.h_conv1)
        print(f'h_pool1: {self.h_pool1.get_shape()}')

        # second convolutional layer
        self.w_conv2 = xavier_init([1, 25, 32, 64])
        self.b_conv2 = bias_variable([64])

        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.w_conv2) + self.b_conv2)
        print(f'h_conv2: {self.h_conv2.get_shape()}')
        self.h_pool2 = max_pool_2x2(self.h_conv2)
        print(f'h_pool2: {self.h_pool2.get_shape()}')

        # densely connected layer
        self.w_fc1 = xavier_init([1 * 500 * 64, 1024])
        self.b_fc1 = bias_variable([1024])

        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 1 * 500 * 64])
        print(f'h_pool2_flat: {self.h_pool2_flat.get_shape()}')
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.w_fc1) + self.b_fc1)
        print(f'h_fc1: {self.h_fc1.get_shape()}')

        # dropout
        self.keep_prob = tf.placeholder("float")
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        # readout layer
        self.w_fc2 = xavier_init([1024, CLASS_NUM])
        self.b_fc2 = bias_variable([CLASS_NUM])

        # From Site1997:
        # This would cause nan or 0 gradient if "tf.matmul(h_fc1_drop, w_fc2) + b_fc2" is all zero or nan,
        # so when the training iteration is big enough, all weights could suddenly became 0.
        # Use tf.nn.ax_cross_entropy_with_logits instead. It handles the extreme case safely.
        self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.w_fc2) + self.b_fc2)
        print(f'y_conv: {self.y_conv.get_shape()}')
        
        # y_conv = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

        # define var&op of training&testing
        self.actual_label = tf.argmax(self.y_, 1)
        self.label, self.idx, self.count = tf.unique_with_counts(self.actual_label)
        self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y_conv))
        #self.train_step = tf.train.GradientDescentOptimizer(1e-8).minimize(self.cross_entropy)
        self.train_step = tf.train.AdamOptimizer(5e-10).minimize(self.cross_entropy)
        #self.train_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(self.cross_entropy)
        #self.train_step = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(self.cross_entropy)
        self.predict_label = tf.argmax(self.y_conv, 1)
        self.label_p, self.dx_p, self.count_p = tf.unique_with_counts(self.predict_label)
        self.correct_prediction = tf.equal(self.predict_label, self.actual_label)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        self.correct_label = tf.boolean_mask(self.actual_label, self.correct_prediction)
        self.label_c, self.idx_c, self.count_c = tf.unique_with_counts(self.correct_label)

    def __del__(self):
        self.sess.close()

    def fit(self, dataset, y=None, model_dir=""):
        # if model exists: restore it
        # else: train a new model and save it
        saver = tf.train.Saver()
        model_name = f"model_{str(CLASS_NUM)}class_{model_dir}"
        model = model_name + "/" + model_name + ".ckpt"
        print(model)
        i = 0
        if not os.path.exists(model + ".meta"):
            self.sess.run(tf.global_variables_initializer())
            if not os.path.exists(model_name):
                os.makedirs(model_name)

            if y is not None:
                dataset = input_data.DataSet(dataset, input_data.dense_to_one_hot(y, num_classes=3), skip_reshape=True)

            for i in range(TRAIN_ROUND + 1):
                batch = dataset.next_batch(1) #50
                if i % 5 == 0:  #100
                    print(self.accuracy)
                    train_accuracy = self.accuracy.eval(
                        feed_dict={
                            self.x: batch[0],
                            self.y_: batch[1],
                            self.keep_prob: 1.0,
                        }
                    )
                    s = "step %d, train accuracy %g" % (i, train_accuracy)
                    print(s)
                    print(f'batch[1] : {batch[1]}')
                    # if i%2000 == 0:
                    #     with open('out.txt','a') as f:
                    #         f.write(s + "\n")
                self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
                # i += 1

            # save_path = saver.save(self.sess, model)
            # print("Model saved in file:", save_path)
        else:
            saver.restore(self.sess, model)
            print("Model restored: " + model)

    def predict(self, X):
        # Ensure X has the correct shape
        print(f'Input X shape: {X.shape}')  # Added
        X = X.reshape((-1, 1500 * 1500))  # Added
        print(f'Reshaped X shape: {X.shape}')  # Added
        params = [self.y_conv]
        y = self.sess.run(
            params,
            {self.x: X, self.keep_prob: 1.0},
        )
        print(f'params : {params}')
        print(f'x : {self.x.get_shape()}')
        print(f'Output y shape: {len(y[0])}, y contents: {y[0]}')  # Added

        return np.array([np.argmax(test) for test in y[0]])

    def predict_proba(self, X):
        params = [self.y_conv]
        y = self.sess.run(
            params,
            {self.x: X, self.keep_prob: 1.0},
        )

        return np.array(y[0])