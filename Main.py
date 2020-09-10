import pandas as pd
import numpy as np
from laspy.file import File

from Mask import Mask

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
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(data, data_targets, test_size = 0.1)

print(xTrain.shape)

'''
Define the model
'''
# import tensorflow as tf
import keras
from keras import layers
from keras.layers import Conv2D, InputLayer, UpSampling2D, MaxPooling2D, Dense, Flatten, Conv3D
from keras.models import Sequential


# block 1
inputs=keras.Input(shape=(None,3))
x=Conv2D(2,2)(inputs)
x=MaxPooling2D(2,2)(x)

x=Flatten(input_shape=2)(x)
block_1_output=Dense(256)(x)


# outputs
outputs =Dense(7, activation='sigmoid')(block_1_output)


# image of model
model=keras.Model(inputs=inputs, outputs=outputs, name="sem_seg_model")
keras.utils.plot_model(model, "./data/sem_seg_model.png", show_shapes=True)

model.summary()


# compile model
model.compile(optimizer='rmsprop',loss='mse')

# there is an issue fitting the data
model.fit(x=xTrain,y=yTrain, batch_size=256,verbose=1, epochs=100)
