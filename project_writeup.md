# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./project_pics/example.jpg "Example Image"
[image2]: ./project_pics/flipped.jpg "Flipped Image"
[image3]: ./project_pics/croppedjpg "Cropped Image"
[image4]: ./project_pics/y_channel.jpg "Y Channel Image"
[image5]: ./project_pics/u_channel.jpg "U Channel Image"
[image6]: ./project_pics/v_channel.jpg "V Channel Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* project_writeup.md summarizing the results
* run1.mp4 containing the video of the car being driven autonomously

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
An example of the car driving can be seen in the video `run1.mp4`.

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network with layers for cropping the image (line 105) and normalizing values (line 106), 3 2d convolutional layers with 5x5 kernel sizes and 8, 16, and 32 features respectively, and 2 3x3 2d convolution layers each with 64 features. The network can be seen in code lines 105 to 125.

The model includes RELU layers to introduce nonlinearity (seen as the activation option for the convolutional layers). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on the Udacity dataset combined with data collected from my own driving in the simulator. The data set was made twice as large by reversing the image and angle data for all images and adding them to the data set. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (clone.py line 125).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, with a special focus on recording runs around curves and accross bridges. Left, Center, and Right camera angles were used for the training set. The colorspace of the dataset was transformed to the YUV color space.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with the LeNet architecture and start changing and adding convolutional layers and retraining.

My first step was to use LeNet, which is tried and tested. I then tried a convolution neural network model similar to the model I used in project 2. I thought this model might be appropriate because it was based on LeNet and trained on numerous images, and I would be retraining the model from scratch anyway. I looked up examples of other popular architectures like VGG to see their structures and inform my decisions.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. Adding much more data by combining the Udacity dataset and increasing batchsize a bit and shuffling data helped with this dramatically.

Much of what improved my model was how I modified my dataset. One of the last major improvements before my care was able to drive around the track on it's own was increasing the amount of data and using the YUV colorspace.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, particularly where the car had the make the turn near the dirt road and occasionally around other corners. To improve the driving behavior in these cases, I took more recordings of me driving around this areas.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (clone.py lines 105-125) consisted of a convolution neural network with the following layers and layer sizes:

**Layer Type** | **Shape**
--- | ---
Input | (160, 320, 3)
Cropping | (90, 320, 3)
Normalization | (90, 320, 3)
Convolutional [5x5 kernel, valid, 2x2 step] | (43, 158, 8)
Activation [ReLU] | (43, 158, 8)
Convolutional [5x5 kernel, valid, 2x2 step] | (20, 77, 16)
Activation [ReLU] | (20, 77, 16)
Convolutional [5x5 kernel, valid, 2x2 step] | (8, 37, 32)
Activation [ReLU] | (8, 37, 32)
Convolutional [3x3 kernel, valid, 1x1 step] | (6, 35, 64)
Activation [ReLU] | (6, 35, 64)
Convolutional [3x3 kernel, valid, 1x1 step] | (4, 33, 64)
Activation [ReLU] | (4, 33, 64)
Fully Connected | (8448)
Fully Connected | (120)
Fully Connected | (60)
Fully Connected | (30)
Fully Connected | (1)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

![alt text][image2]

![alt text][image3]

![alt text][image4]

![alt text][image5]

![alt text][image6]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
