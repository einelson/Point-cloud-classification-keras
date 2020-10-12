# Point-cloud-classification-keras
Point Net++ for semantic segmentation on .las files

Mask.py is used to correctly import the .las files
Origin.py is the original file found on: https://keras.io/examples/vision/pointnet/
pointnet.py is the edited file that is able to take in .las files correctly preprocess the data and train on the data

The neural network used in training does not use the whole input file, instead it traines on a sample of data points. You can change the sample size by changing the NUM_POINTS var
I have not figured out a way to visualize the results yet

The .las files are to be put into the data folder in their respective sub folders.
There can be multiple files per folder
