__author__ = 'Shane Licari'
__email___ = "shanelicari28@gmail.com"
__date__ = "07/17/2020"

# Python packages
import os
import csv
import socket
import ray
import psutil
import itertools
import pandas as pd
import numpy as np
import cv2

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from sklearn.model_selection import train_test_split
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # Dynamically grow the memory used on the GPU
config.log_device_placement = True  # Log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

##########################################
# Get training data from csv files
##########################################

def get_data():
    # Hardcoded paths for simulation data
    simulation_data_path = './simulation_data'
    csv_data_filename = 'driving_log.csv'

    lines = []

    # Get all data folder paths
    data_flds = os.listdir(simulation_data_path)

    # Loop through simluation data folders
    for fld in data_flds:
        # Create path to csv data
        data_path = os.path.join(simulation_data_path, fld, csv_data_filename)

        # Open csv file
        with open(data_path) as csvfile:
            print("Opening data file --> {}".format(data_path))
            # Read data from csv
            reader = csv.reader(csvfile)

            # Grab each line of csv and append to list
            for line in reader:
                lines.append(line)

    print("Found {} images for training!".format(len(lines)*3))
    return lines


##########################################
# Store training data in pandas dataframe
##########################################
def to_df(lines):
    # Dataframe column names
    df_columns = ['center', 'left', 'right', 'angle', 'throttle', 'brake', 'speed']

    # Put training data into dataframe
    t_data_df = pd.DataFrame(lines, columns=df_columns)

    # Convert angle column into a float
    t_data_df['angle'] = t_data_df['angle'].astype('float32')

    #############################################
    # Fix all image paths from training data
    # because it was collected on diff PC's
    #############################################
    # Convert any windows type paths to linux
    t_data_df['center'] = t_data_df.center.str.replace(r'\\', '/')
    t_data_df['right'] = t_data_df.right.str.replace(r'\\', '/')
    t_data_df['left'] = t_data_df.left.str.replace(r'\\', '/')

    # Get host name of PC
    host_name = socket.gethostname()

    if host_name == 'had-HP-Z4-G4-Workstation':
        t_data_df['center'] = t_data_df.center.str.replace('/home/shane/dev/udacity_self_driving_car_engineer/',
                                                           '/home/slicari/dev/udacity/')
        t_data_df['right'] = t_data_df.right.str.replace('/home/shane/dev/udacity_self_driving_car_engineer/',
                                                         '/home/slicari/dev/udacity/')
        t_data_df['left'] = t_data_df.left.str.replace('/home/shane/dev/udacity_self_driving_car_engineer/',
                                                       '/home/slicari/dev/udacity/')

        t_data_df['center'] = t_data_df.center.str.replace(r'C:/', '/home/slicari/')
        t_data_df['right'] = t_data_df.right.str.replace(r'C:/', '/home/slicari/')
        t_data_df['left'] = t_data_df.left.str.replace(r'C:/', '/home/slicari/')

    return t_data_df

##########################################
# Function to be executed by ray workers
##########################################
@ray.remote
def image_preprocess(sample, validation_flag):
    images = []
    angles = []
    augmented_images, augmented_angles = [], []

    # Open center, left and right images
    center_image = cv2.cvtColor(cv2.imread(sample[0]), cv2.COLOR_BGR2RGB)
    left_image = cv2.cvtColor(cv2.imread(sample[1]), cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(cv2.imread(sample[2]), cv2.COLOR_BGR2RGB)

    # Cropt out the top and bottom of the image to remove the sky
    # and hood of the car so the model doesn't consider these features while training
    center_image = center_image[50:140, ]
    left_image = left_image[50:140, ]
    right_image = right_image[50:140, ]

    # Get steering angle
    angle_center = sample[3]

    # Apply steering angle correction for left and right image
    correction = 0.2
    angle_left = angle_center + correction
    angle_right = angle_center - correction

    # Append image to list
    images.extend((center_image, left_image, right_image))

    # Append steering angle to list
    angles.extend((angle_center, angle_left, angle_right))

    for image, angle in zip(images, angles):
        augmented_images.append(image)
        augmented_angles.append(angle)

        # Augment training data no validation data
        if validation_flag == False:
            # Flip image horizontally
            augmented_images.append(cv2.flip(image, 1))

            # Flip steering angle
            augmented_angles.append(angle * -1)

            # Randomly adjust brightness of image
            aug_img = image.copy()

            # Get random value to adjust brightness
            b_val = np.random.randint(-25, 25)

            # Adjust pixel values by b_val and clip range 0-255
            aug_img[:, :, :] = np.clip(aug_img[:, :, :] + b_val, 0, 255)

            augmented_images.append(aug_img)
            augmented_angles.append(angle)

    return (augmented_images, augmented_angles)

##########################################
# Function to be balance the training set
##########################################
def flatten_dataset(X_train, y_train):
    num_bins = 50
    avg_samples_per_bin = len(y_train)/num_bins
    bin_threshold = int(avg_samples_per_bin) + 500
    bin_counts, bins = np.histogram(y_train, num_bins)

    keep_val = []

    for i in range(num_bins):
        if bin_counts[i] < bin_threshold:
            keep_val.append(1.)
        else:
            keep_val.append(1./(bin_counts[i]/bin_threshold))

    remove_list = []
    i = 0
    for i in range(len(y_train)):
        for j in range(num_bins):
            if y_train[i] > bins[j] and y_train[i] <= bins[j+1]:
                if np.random.rand() > keep_val[j]:
                    remove_list.append(i)

    y_train = np.delete(y_train, remove_list)
    X_train = np.delete(X_train, remove_list, axis=0)

    return X_train, y_train


############################################
# CNN Architecture
############################################
def nvidia_model():
    # """
    # Modified NVIDIA model
    # """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(90, 320, 3)))
    #model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()

    return model

def main():
    # Get data from all csv files in simulation data folder
    lines = get_data()

    # Store data in pandads df and fix image paths
    data_df = to_df(lines)

    # Put each row of dataframe into list
    samples = data_df.values.tolist()

    # Get number of available cpu cores
    num_cpus = psutil.cpu_count(logical=True)

    # Initialize Ray
    ray.init(num_cpus=num_cpus)

    augmented_data = []
    validation_data = []

    # Split data into training and validation sets
    train_samples, validation_samples = train_test_split(samples, test_size=0.2, random_state=832289)
    
    # Let start parallel processing with Ray!
    for sample in train_samples:
        augmented_data.append(image_preprocess.remote(sample, validation_flag=False))

    for sample in validation_samples:
        validation_data.append(image_preprocess.remote(sample, validation_flag=True))

    train_results = ray.get(augmented_data)
    valid_results = ray.get(validation_data)

    data = list(zip(*ray.get(augmented_data)))
    x_train_lst = list(itertools.chain(*data[0]))
    y_train_lst = list(itertools.chain(*data[1]))

    data = list(zip(*ray.get(validation_data)))
    x_valid_lst = list(itertools.chain(*data[0]))
    y_valid_lst = list(itertools.chain(*data[1]))

    X_train = np.array(x_train_lst)
    y_train = np.array(y_train_lst)

    X_valid = np.array(x_valid_lst)
    y_valid = np.array(y_valid_lst)

    X_train, y_train = flatten_dataset(X_train, y_train)

    assert(X_train.shape[0] == y_train.shape[0])

    # Load model
    model = nvidia_model()

    #parallel_model = multi_gpu_model(model, gpus=2)

    # Compile model
    model.compile(loss='mse', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

    # Train model
    batch_size=128
    history_object = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=5, batch_size=batch_size)

    # Save model
    model.save('model.h5')

if __name__ == "__main__":
    main()




