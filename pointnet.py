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
def preprocess_las(data):
    data_targets=data

    # first 3 columns are the x values
    data=data.drop(['r', 'g', 'b', 'class'], axis=1)

    # the last column is the y value
    data_targets = data_targets.drop(['x', 'y', 'z', 'r', 'g', 'b'], axis=1)
    # print(new_df['class'].unique())


    # Reshape data to correct format
    # print(data.shape) -> (17848546, 3)
    
    data=np.array(data)
    data=np.expand_dims(data, axis=0)  # shape -> (1, 17848546, 3)
    


    # print(data_targets.shape) -> (17848546, 1)
    labels = (np.array(data_targets).flatten()) # shape -> (17848546,)
    # print(labels)    


    return data, labels


"""
To generate a `tf.data.Dataset()` we need to first parse through the ModelNet data
folders. Each mesh is loaded and sampled into a point cloud before being added to a
standard python list and converted to a `numpy` array. We also store the current
enumerate index value as the object label and use a dictionary to recall this later.
"""


def parse_dataset(num_points=2048):
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}


    # open files - will need to loop through all data
    train_df=open_las('./data/Room1_filtered.las')
    test_df=open_las('./data/Room1_filtered.las')

    # prepreocess
    train_points, train_labels = preprocess_las(train_df)
    test_points, test_labels = preprocess_las(test_df)


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
NUM_POINTS = 65,536
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


# print(train_points.shape) #(3126, 2048, 3) (number of files loaded, batch size, XYZ_value)
# print(train_labels.shape) #(3126,) (number of files loaded where each file is a class)
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

model.fit(train_dataset, epochs=20, validation_data=test_dataset)

"""
## Visualize predictions
We can use matplotlib to visualize our trained model performance.
"""

data = test_dataset.take(1)

points, labels = list(data)[0]
points = points[:8, ...]
labels = labels[:8, ...]

# run test data through model
preds = model.predict(points)
preds = tf.math.argmax(preds, -1)

points = points.numpy()

# plot points with predicted class and label
fig = plt.figure(figsize=(15, 10))
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
    ax.set_title(
        "pred: {:}, label: {:}".format(
            CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
        )
    )
    ax.set_axis_off()
plt.show()