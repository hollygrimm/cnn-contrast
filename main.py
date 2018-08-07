from __future__ import print_function
import os
import time
import logging
import argparse
import tensorflow as tf
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import cv2
import pandas as pd
from PIL import Image

from models.model import Model

IMG_SIZE = 64  # FIXME pass in, 256 on the short side
BATCH_SIZE = 2 # FIXME pass in
train_test_csv_file = "data/all_domain.csv"

def config_logging(log_file):
    tf.logging.set_verbosity(tf.logging.INFO)
    if os.path.exists(log_file):
        os.remove(log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def cnn_model_fn(features, labels, mode):

    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, IMG_SIZE, IMG_SIZE, 3])

    input_layer = input_layer/255

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]

    pool2_flat = tf.contrib.layers.flatten(pool2)

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense2 = tf.layers.dense(inputs=dropout, units=100, activation=tf.nn.relu)

    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)    

    # Input Tensor Shape: [batch_size, 100]
    # Output Tensor Shape: [batch_size, 1]
    y = tf.layers.dense(inputs=dropout2, units=1) # no activation

    y = tf.reshape(y, [-1], name="y")

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "contrast": y,
    }   

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate MSE Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels, y)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    # FIXME: change accuracy metric here
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["contrast"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def create_train_test_data():
    # LABEL_FIELD = "variety_texture"
    # LABEL_FIELD = "variety_shape"
    # LABEL_FIELD = "variety_size"
    #LABEL_FIELD = "variety_color"
    # LABEL_FIELD = "variety"
    LABEL_FIELD = "contrast"
    # LABEL_FIELD = "repetition"
    # LABEL_FIELD = "unity"
    # LABEL_FIELD = "balance"
    df = pd.read_csv(train_test_csv_file, usecols=['in_train', 'new_filename', LABEL_FIELD])
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []    

    filtered = df.dropna(subset=[LABEL_FIELD])

    for in_train, new_filename, LABEL_FIELD in filtered.values:
        if in_train:
            img_file_path = os.path.join('data/train', new_filename)
            img_data = np.asarray(Image.open(img_file_path).resize((IMG_SIZE, IMG_SIZE)).convert('RGB'), dtype='float32')
            # n_white_pix = np.sum(img_data == 255)
            # n_black_pix = np.sum(img_data == 0)
            # print("n white pix: {}".format(n_white_pix))
            # print("n black pix: {}".format(n_black_pix))
            train_data.append(img_data)
            train_labels.append(LABEL_FIELD)
        else:
            img_file_path = os.path.join('data/test', new_filename)
            img_data = np.asarray(Image.open(img_file_path).resize((IMG_SIZE, IMG_SIZE)).convert('RGB'), dtype='float32')
            test_data.append(img_data)
            test_labels.append(LABEL_FIELD)

    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)



def create_label(image_name):
    label = image_name.split('.')[-2]
    return int(label)

def create_train_data():
    train_data = []
    train_labels = []

    for img_filename in tqdm(os.listdir('data/train')):
        im_file_path = os.path.join('data/train', img_filename)       
        img_data = np.asarray(Image.open(im_file_path).resize((IMG_SIZE, IMG_SIZE)).convert('RGB'), dtype='float32')
        # images[given_im,:,:,:] = im
        train_data.append(img_data)
        train_labels.append(create_label(img_filename))

    shuffle(train_data)
    return np.array(train_data), np.array(train_labels)

def create_test_data():
    test_data = []
    test_labels = []
    for img_filename in tqdm(os.listdir('data/test')):
        im_file_path = os.path.join('data/test', img_filename)          
        img_data = np.asarray(Image.open(im_file_path).resize((IMG_SIZE, IMG_SIZE)).convert('RGB'), dtype='float32')
        # images[given_im,:,:,:] = im   
        test_data.append(img_data)
        test_labels.append(create_label(img_filename))
    shuffle(test_data)
    return np.array(test_data), np.array(test_labels)


def train(learning_rate=1e-3,
          results_dir=None,
          checkpoint_dir=None,
          num_layers=1,
          layer_size=200,
          batch_size=10,
          restore=False,
          optimizer='adam',
          keep_prob=1,
          ):

    # train_data, train_labels = create_train_data()
    # eval_data, eval_labels = create_test_data()
    train_data, train_labels, eval_data, eval_labels = create_train_test_data()

    # Create the Estimator
    regressor = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/cnn_contrast_model") # TODO: change folder

    # Set up logging for predictions
    # Log the values in the "y" tensor with label "values"
    tensors_to_log = {"values": "y"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=10000,
        shuffle=True)
    regressor.train(
        input_fn=train_input_fn,
        steps=1000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = regressor.evaluate(input_fn=eval_input_fn)

    predictions = list(regressor.predict(input_fn=eval_input_fn))
    predicted_values = [p["contrast"] for p in predictions]
    confusion_matrix = tf.confusion_matrix(eval_labels, predicted_values)

    sess = tf.Session()
    with sess.as_default():
        print(eval_labels)
        print(predicted_values)
        print(confusion_matrix.eval())

    print("eval metrics: %r"% eval_results)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--exp_name', type=str, default='cnn')
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--layer_size', '-s', type=int, default=200)
    parser.add_argument('--batch_size', '-b', type=int, default=10)
    parser.add_argument('--restore', '-restore', action='store_true')
    args = parser.parse_args()

    checkpoint_dir = os.path.join(os.getcwd(), 'results')
    results_dir = os.path.join(
        os.getcwd(), 'results', args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S"))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    train(
        learning_rate=args.learning_rate,
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
        num_layers=args.n_layers,
        layer_size=args.layer_size,
        batch_size=args.batch_size,
        restore=args.restore
    )


if __name__ == "__main__":
    log_file = os.path.join(os.getcwd(), 'results', 'train_out.log')
    logger = config_logging(log_file)

    main()
