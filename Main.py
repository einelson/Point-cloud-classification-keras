import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from laspy.file import File

from Mask import Mask

'''
Define conv functions
'''
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



def conv_bn(x, filters):
    x = keras.layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = keras.layers.BatchNormalization(momentum=0.0)(x)
    return keras.layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = keraslayers.Dense(filters)(x)
    x = keras.layers.BatchNormalization(momentum=0.0)(x)
    return keras.layers.Activation("relu")(x)

def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = keras.layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])






'''
Open the .las file
'''
las_header = None
max_points=1000000000
f = File('./data/Room1_filtered.las')

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



'''
Preprocess the data
'''
# first 3 columns are the x values
data=new_df
data=data.drop(['r', 'g', 'b', 'class'], axis=1)

# the last column is the y value
data_targets=new_df
data_targets = data_targets.drop(['x', 'y', 'z', 'r', 'g', 'b'], axis=1)
# print(new_df['class'].unique())

# split the data
xTrain, xTest, yTrain, yTest = train_test_split(data, data_targets, test_size = 0.1)


yTrain = keras.utils.to_categorical(yTrain)
yTest = keras.utils.to_categorical(yTest)

# print(yTrain)
'''
Define the model
'''




# block 1
inputs=keras.Input(shape=(3,))
# block_1_output=Dense(256)(inputs)
# # outputs
# outputs =Dense(8, activation='softmax')(block_1_output)

x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = keras.layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = keras.layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(8, activation='softmax')(x)

# image of model
model=keras.Model(inputs=inputs, outputs=outputs, name="sem_seg_model")
keras.utils.plot_model(model, "./data/sem_seg_model.png", show_shapes=True)

model.summary()


# compile model
model.compile(optimizer='rmsprop',loss='mse', metrics=['accuracy'])

# there is an issue fitting the data
model.fit(x=xTrain,y=yTrain, batch_size=2040,verbose=1, epochs=5, validation_data=(xTest,yTest))
score, acc = model.evaluate(x=xTest, y=yTest)

print('Accuracy: ',100*(acc))