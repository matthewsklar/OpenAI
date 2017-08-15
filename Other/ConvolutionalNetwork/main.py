import tensorflow as tf
import numpy as np

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python import SKCompat

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    # Input Layer
    input = tf.reshape(features, [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input,
        filters=32,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu
    )

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=5,
        padding='same',
        activation=tf.nn.relu
    )

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    loss = None
    train_op = None

    # Calculate loss for both TRAIN and EVAL modes
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer='SGD'
        )

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def input_fn(data_set, batch_size=100, seed=None, epochs_n=1):
    np_labels = np.asarray(data_set.labels, dtype=np.int32)

    all_images = tf.constant(data_set.images, shape=data_set.images.shape, verify_shape=True)
    all_labels = tf.constant(np_labels, shape=np_labels.shape, verify_shape=True)

    image, label = tf.train.slice_input_producer(
        [all_images, all_labels],
        num_epochs=epochs_n,
        shuffle=(seed is not None), seed=seed,
    )

    dataset_dict = dict(images=image, labels=label)  # This becomes pluralized into batches by .batch()

    batch_dict = tf.train.batch(
        dataset_dict, batch_size,
        num_threads=1, capacity=batch_size * 2,
        enqueue_many=False, shapes=None, dynamic_pad=False,
        allow_smaller_final_batch=False,
        shared_name=None, name=None
    )

    return batch_dict['images'], batch_dict['labels']


def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')

    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = learn.Estimator(model_fn=cnn_model_fn, model_dir='/tmp/mnist_convnet_model')

    # Set up logging for predictions
    tensor_to_log = { 'probabilities': 'softmax_tensor' }
    logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=50)

    # Train the model
    '''mnist_classifier.fit(
        x=train_data,
        y=train_labels,
        batch_size=100,
        steps=20000,
        monitors=[logging_hook]
    )'''
    mnist_classifier.fit(
        input_fn=lambda: input_fn(mnist.train, seed=42),
        steps=2000,
        monitors=[logging_hook]
    )

    # Configure the accuracy metric for evaluation
    metrics = {
        'accuracy': learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key='classes')
    }

    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(x=eval_data, y=eval_labels, metrics=metrics)

    print(eval_results)

if __name__ == '__main__':
    tf.app.run()