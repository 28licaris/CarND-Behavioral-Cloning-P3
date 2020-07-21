# **Udacity Self-Driving Car Engineer Nanodegree - Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/center_2020_06_28_09_24_50_500.jpg
[image2]: ./writeup_images/left_2020_06_28_09_24_50_500.jpg
[image3]: ./writeup_images/right_2020_06_28_09_24_50_500.jpg
[image4]: ./writeup_images/df_data.png
[image5]: ./writeup_images/center_image_bef_crop.png
[image6]: ./writeup_images/center_image_aft_crop.png
[image7]: ./writeup_images/before_flatten.png
[image8]: ./writeup_images/after_flatten.png

---

#### 1. Submission files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* CarND-Behavioral-Cloning.ipynb used for developing pipeline in model.py

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Creation of the Training Set 

##### Data Collection
To first get my pipeline working I recorded 2 laps on track one clockwise and the other counterclockwise. I made sure the vehicle was in the center of the road as much as possible while collecting this training data. I also used the mouse for steering as suggested. Below you can see images collected from the left, center and right camera on track 1.

![alt text][image2] ![alt text][image1] ![alt text][image3]

Once I got my pipeline working with the first 2 laps of data I decided to collect a lot more data. I collected 2 more laps on track one driving in the center both clockwise and counter clockwise. I also added some correction training data by recording the car recover from the side of the road to the center of the lane. I also added 2 laps from track two. One lap was driving in reverse and I kept the car in the center of the road for both of these laps. The training data collected is shown below.
![alt text][image4]

My final collected training set has a total of 19578 images before data preprocessing and augmentation was applied

##### Data Preprocessing and Augmentation
To create a training and validation set I used the following code:

`# Create training samples and validation samples
train_samples, validation_samples = train_test_split(samples, test_size=0.2)`


I chose to use 20% of the data for validation which left me with 15660 samples for training and 3918 for validation.

Next I applied the image processing pipeline to both datasets with the exception that I did not augment the validation set.

First the image was cropped to remove features in the image the model should not consider like the  sky, trees, etc.

###### Before Crop
![alt text][image5]
###### After Crop
![alt text][image6]

After cropping the image I augmented the data by flipping images horizontally and also creating an image with the brightness randomly adjusted. I used Ray for parallel processing as shown below.

```
##########################################
# Function to be executed by ray workers
##########################################
@ray.remote
def image_preprocess(sample):
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

```

After creating an augmented training set I was left with the following data distribution.
![alt text][image7]

As you can see the data isn't very uniformly distributed across the steering angles so I chose to flatten the distribution by only allowing each bin to have at max 1752 images. You can find how the data is flattened in the flatten_dataset() function in model.py. I refered to this repo https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project/blob/master/model.py for the flatten_dataset() function. After using the flatten_dataset() function we get the dataset shown in orange which will be fed into the model for training:

![alt text][image8]

The final training set with augmented images contains 33515 images.
The final validation set contains 3918 images.

#### Training
For training I loaded all of the data into memory becuase memory was not an issue on my local machine. I also implemented a training generator as well but the model.py final pipeline loads all the data into memory. 

I used an adam optimizer so I did not manually tune the learning rate and the final model architecture shown below was trained for 5 epochs.

#### 4. Model Architecture

At first I implemented the LeNet architecture in Keras because I was familar with tha architecture from the previous project. I used this to validate my pipeline was working and that I could get get the car to somewhat stay on the track. After getting the car to almost make it around corners I chose to collect more data and use a slightly modified version of the NVIDIA architecture. Info about this architecture can be found here https://developer.nvidia.com/blog/deep-learning-self-driving-cars/. 

#### Final Model Architecture
My final model consists of using the NVIDIA model as suggested for this project with a few modifications. The input shape for the image after cropping is 90x320x3 using RGB color space instead of YUV. I used 'relu' for the activation function between layers. I also added in dropout layers to help prevent overfitting.

`
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_4 (Lambda)            (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 4, 33, 64)         36928     
_________________________________________________________________
flatten_2 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 8448)              0         
_________________________________________________________________
dense_5 (Dense)              (None, 100)               844900    
_________________________________________________________________
dropout_5 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_6 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_6 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_7 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_8 (Dense)              (None, 1)                 11  
_________________________________________________________________
Total params: 981,819
`


#### 5. Conclusion

Overall this project was very rewarding. It truly shows the power of CNN's and the kind of tasks they can perform without explicity programming. I think the biggest challenge for this project was getting good training data and preprocessing it sufficiently. I ended up removing a lot of the training data collected to make the dataset look more evenly distributed across steering angles. I think an improvement could be adding more data augmentatoin as well as trying to get a gaussian distribution for the dataset. I did not include it in this submission but I was also able to remove the PID controller and train the vehicle on the throttle input so it can drive itself around the track completely. This projet was a lot of fun and is a good excercise going through the complete process of data collection, preprocessing, training, and testing. Below is a link to my final video submission of the car driving around track 1 using the model I trained. 

https://drive.google.com/file/d/1q4PnYRUD1LzlybFc-0ao-WW6urYgQQe5/view?usp=sharing
