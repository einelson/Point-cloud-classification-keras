"""
Title: Point cloud classification with PointNet for warehouse
Author: [David Griffiths](https://dgriffiths3.github.io)
Modified by: Ethan Nelson
Date created: 2020/05/25
Last modified: 09/23/2020
Description: Implementation of PointNet for warehouse classification.

Introduction:
Classification, detection and segmentation of unordered 3D point sets i.e. point clouds
is a core problem in computer vision. This example implements the seminal point cloud
deep learning paper [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593). For a
detailed intoduction on PointNet see [this blog
post](https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a).

This code is modified from https://github.com/keras-team/keras-io/blob/master/examples/vision/pointnet.py
"""

"""
## Setup
"""


import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import pandas as pd

from Mask import Mask
from laspy.file import File

tf.random.set_seed(1234)

"""
To open las files and returnteh test and training points
"""
def open_las(file='none'):
    if file =='none':
        print('No files selected')
        return


    las_header = None
    max_points=1000000000
    f = File(file)

    if las_header is None:
        las_header = f.header.copy()
    if max_points is not None and max_points < f.header.point_records_count:
        mask = Mask(f.header.point_records_count, False)
        mask[np.random.choice(f.header.point_records_count, max_points)] = True
    else:
        mask = Mask(f.header.point_records_count, True)
        new_df = pd.DataFrame(np.array((f.x, f.y, f.z)).T[mask.bools])
        new_df.columns = ['x', 'y', 'z']
    if f.header.data_format_id in [2, 3, 5, 7, 8]:
        rgb = pd.DataFrame(np.array((f.red, f.green, f.blue), dtype='int').T[mask.bools])
        rgb.columns = ['r', 'g', 'b']
        new_df = new_df.join(rgb)
    new_df['class'] = f.classification[mask.bools]
    if np.sum(f.user_data):
        new_df['user_data'] = f.user_data[mask.bools].copy()
    if np.sum(f.intensity):
        new_df['intensity'] = f.intensity[mask.bools].copy()
    
    return new_df


'''
Preprocess the las data and reshape
'''
def preprocess_las(data, sample_points=2048):
    data_targets=data

    # first 3 columns are the x values, get rid of everything else
    data=data.drop(['r', 'g', 'b', 'class'], axis=1)
    
    data=np.array(data)

    data=data[:sample_points, :]
    return data#, labels


"""
To generate a `tf.data.Dataset()` we need to first parse through the ModelNet data
folders. Each mesh is loaded and sampled into a point cloud before being added to a
standard python list and converted to a `numpy` array. We also store the current
enumerate index value as the object label and use a dictionary to recall this later.
"""


def parse_dataset(NUM_POINTS=2048):
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}

    import glob
    folders=glob.glob('./data/*')
    
    # iterate through the classes
    for i, folder in enumerate(folders):
        # class_map[i] = folder.split("/")[-1]

        print("processing class: {}".format(os.path.basename(folder)))

        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(preprocess_las(open_las(f), NUM_POINTS))
            train_labels.append(i)


        for f in test_files:
            test_points.append(preprocess_las(open_las(f), NUM_POINTS))
            test_labels.append(i)


    return (
        np.array(train_points), # (3126, 2048, 3) (number of files(usually each file contains 1 class), points from each file (batch size, not the whole file size), XYZ coordinates)
        np.array(test_points),  # (636, 2048, 3)
        np.array(train_labels), # (3126,) (number of files loaded where each file is a class)
        np.array(test_labels),  # (636,)
        class_map,
    )


"""
Set the number of points to sample and batch size and parse the dataset. This can take
~5minutes to complete.
"""

# NUM_POINTS = 2048
NUM_POINTS = 30000
NUM_CLASSES = 10
BATCH_SIZE = 32

train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
    NUM_POINTS
)

"""
Our data can now be read into a `tf.data.Dataset()` object. We set the shuffle buffer
size to the entire size of the dataset as prior to this the data is ordered by class.
Data augmentation is important when working with point cloud data. We create a
augmentation function to jitter and shuffle the train dataset.
"""


def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

"""
### Build a model
Each convolution and fully-connected layer (with exception for end layers) consits of
Convolution / Dense -> Batch Normalization -> ReLU Activation.
"""


def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


"""
PointNet consists of two core components. The primary MLP network, and the transformer
net (T-net). The T-net aims to learn an affine transformation matrix by its own mini
network. The T-net is used twice. The first time to transform the input features (n, 3)
into a canonical representation. The second is an affine transformation for alignment in
feature space (n, 3). As per the original paper we constrain the transformation to be
close to an orthogonal matrix (i.e. ||X*X^T - I|| = 0).
"""


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))


"""
 We can then define a general function to build T-net layers.
"""


def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])


"""
The main network can be then implemented in the same manner where the t-net mini models
can be dropped in a layers in the graph. Here we replicate the network architecture
published in the original paper but with half the number of weights at each layer as we
are using the smaller 10 class ModelNet dataset.
"""

inputs = keras.Input(shape=(NUM_POINTS, 3))

x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
model.summary()

"""
### Train model
Once the model is defined it can be trained like any other standard classification model
using `.compile()` and `.fit()`.
"""

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

model.fit(train_dataset, epochs=10, validation_data=test_dataset)

score, acc = model.evaluate(x=train_points, y=train_labels)

print('Accuracy: ', acc*100)
