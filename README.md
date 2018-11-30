# Grey-scale Image Classification using KERAS

## Disclaimer
This is a research project submitted for credit for a course that we just completed. The results seen here are subjective and should not be considered as final or accurate. 

## 1. Introduction
In this study, we try to understand the limits of our system when running a Deep Learning training. The step to train a model is the most time consuming step of the model building process. With contraints put on the hardware, what can we do on the programming side to help us train models better? What if you had a limited amount of time? To try out our hand at augmentation, we will be using the Flickr27 dataset.  

## 2. Objective
Our objectives will include: 
  1. Can we run at least 3 training sessions in a working day (8 hours)? (simple model, little data)
  2. Can we achieve a target accuracy of at least 90%? (complex model, more data)
  3. What is the amount of time the model takes for prediction? (simple model)

These objectives helps us limiting the amount of data we can process and the complexity of the model we can run. Trading of one for the other might help us understand which would provide better value in the long run. 

## 3. Learning so far ...
Let us view some of the points that we have to consider when working with Deep Learning models. 
   1. The number of trainable parameters. Each layer in the model would add more capabilities to the model and possibly help in detecting more features but at the same time would increase the model complexity and therefore take more time to run.
   2. The number of images that the model uses for training and validation. The more (and different) data we have, the model would be able to generalize more accurately. However, running large datasets will make the model run much longer.
   3. The number of epochs we need to reach an acceptable accuracy. The more time, the more accurate. Sometimes to the point of memorizing. A model which reaches its target accuracy in 10 epochs would suggest that the model is very complex for the problem.
This is in no way an exhaustive list but they do constitute some of the most important points that we have to keep in mind. 

Training on a CPU with the parameters provided above proved to be next to impossible. Using the inbuilt GPU improved the training time by a factor of 10 (minimum). 

Regarding the algorithm that we intend on using, we will be testing CNNs (Convolutional Neural Networks). Our intention is to test smaller custom architectures and then move to larger ones. While our code uses a modified version of the InceptionNet v3 architecture, we experimented with others as well and settled for the one with the best performance.

## 4. Software requirements
In terms of software requirements, we will be using the following
  - Python with Keras and Tensorflow
  - GPU drivers and associated installations. Please refer to the link (https://www.tensorflow.org/install/install_windows) to check if and how to install the GPU
  - Most of our programming will happen on Jupyter notebooks rather than python programs, as we will require to view output on a line-by-line execution level. Also Jupyter will help us format the code is a format that is presentable. 
  - Highly suggested (but not mandatory) is installing Anaconda. This will help you create separate environments in which you can execute your projects. Should any of the libraries that we use be upgraded or changed, the failure would be contained within the environment and would not affect all the other developments that you have

## 5. More about the dataset
Flick27 is a collection of 27 classes (such as Apple, Google, McDonald's) with each class containing about 35 images. The rules of Flickr27 state that each picture will contain only one of the 27 logos which have been categorized.The dataset is already broken up into "train" (30 images) and "test" (5 images) sets. When training any model, we need to have a train and validation set, we therefore broke the train set into two sets: a train (24 images) and a validation (6 images). It is best that you put all the image files into sub-folders whose names represent the class to which it belongs. The test set should not be used until you have acceptable training and validation accuracy. 

Each of the class, which had 24 original images, were augmented to 1920 images and the validation set which contained 6 images used similar rules and were augmented to 480 images. This means that we will have 5760 images for training and 1440 images for validation. This will be the start of our test and periodically, we shall reduce the number of images (augmentations) to help us understand the impact of lesser data.  

## 6. System configuration
Laptop with:
  - Windows 10
  - Intel Core i5 - (7th generation) 2.5 GHz
  - NVIDIA GeForce 940MX with 2 GB dedicated VRAM
  - 8 GB DDR4 Memory
  - 1 TB HDD
 
This is close to the minimum requirements necessary to run a small scale image classification project. Standard requirements would include:
  - Intel Core i7 - (7th gernation)
  - NVIDIA GTX Series 970
  - 16 GB DDR4 Memory and 
  - 1 TB HDD
  
I would recommend that you look at Siraj's video that was posted on June 2018. Best Laptop for Machine Learning (https://www.youtube.com/watch?v=dtFZrFKMiPI). And yes, I would highly recommend other videos on Machine Learning posted by Siraj. 
  
## 7. Libraries used
The following are the list of Python Libraries that have been used for this project. All of these libraries can be installed with basic 'pip' commands.

1. numpy
2. pandas
3. skimage
4. openCV (cv2)
5. keras
6. matplotlib -pyplot
7. pathlib
8. h5py
9. os
10. scikitlearn
11. scipy

NOTE: It is highly recommended that you install these libraries within your environment before you run the code files mentioned in section 7. Some of these may already be available with your current python distribution.

## 8. Basic code files
Here is a list of the code files that were used and their functions:
  - CreateModel.ipynb: Your first step is to create a model. There are two ways of creating models. You could import a model programmed in Keras directly (read this link for information on available models https://keras.io/applications/) or you could create your own model. In this case, we will be creating our own model using InceptionV3 as the base. The reason in doing so is that most models work with RGB images only and not with Grey-scale. There are a few variables that you will have to change:
    - Number of channels the image has: 1 represents a Grey-scale image, 3 represents a RGB (or HSV) image
    - Number of classes: This is important as this will represent your final output layer. In this example, the value is set to 3
    
  - TrainModel.ipynb: The next step is to train your model. This step could be the most time consuming process. Remember that this will depend on the system and its configuration that is available. In this example, we ran 100 epochs, each of which took approximately 200 seconds. Notice that in the 67th epoch, we have a training accuracy of 99.97% and a validation accuracy of 98.33%. This does represent an over-fitting problem but only very slightly.
  
  - TestModel.ipynb: Finally, we use the trained model (with weights) and predicted classes for the images that we have in our validation set. The results are not as good as we expected. It was 13 correct predictions out of the 15 available, and this translated to 86.6% accuracy. This might also indicate that the model has started to memorize rather than generalize. 
  
## 9. Conclusion
With a 99.97% training, 98.33% validation, and a 86.66% test, this algorithm does show it is possible to create a highly accurate model with less data. 

Point of note here: The development of this model was for a very specific use case and may not work on all instances of the brand logo. We have found reasonable success during our tests in a very specific and controlled source of new data to test the predictions on. We cannot guarantee that we will get the same levels of accuracies on all instances of the logo in new scenarios.

## 10. What changes would we make?
  1. Find a way to compare images and get a score of the similarity between them. This way we remove duplicates from our train and test sets, thus reducing the training time. This will also give us more space to perhaps even classify a fourth logo.
  2. Change the algorithm to use RGB images instead of Grey-scale images as lose features that are important when converting the images from RGB to Grey-scale.
  3. Find a method of checking what is being detected in the image that is used for prediction. This will help us understand the reasons behind why the classification goes wrong.
  4. Detecting multiple logos in an image. For this algorithm to have any sort of meaningful revenue generation, our next steps should include methods of detecting (and classifying) multiple classes in one image and providing accuracy percentages for each of the detected classes along with bounding boxes.
  
## Citations, Credits, Sources, and References
Filckr27 - Y. Kalantidis, LG. Pueyo, M. Trevisiol, R. van Zwol, Y. Avrithis. Scalable Triangulation-based Logo Recognition. In Proceedings of ACM International Conference on Multimedia Retrieval (ICMR 2011), Trento, Italy, April 2011.

Design Guide for CNN: https://hackernoon.com/a-comprehensive-design-guide-for-image-classification-cnns-46091260fb92 - George Seif
April 2018

Inception Net Design: https://arxiv.org/pdf/1512.00567v3.pdf - Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
Zbigniew Wojna December 2015

Scene Classification with Inception-7: http://lsun.cs.princeton.edu/slides/Christian.pdf - Christian Szegedy, Vincent Vanhoucke, Julian
Ibarz

Understanding how image quality affects Deep neural networks: https://arxiv.org/pdf/1604.04004.pdf - Samuel Dodge, Lina Karam April
2016

Benchmarks for popular CNN models: https://github.com/jcjohnson/cnn-benchmarks - Justin Johnson

Tutorials on CNN: http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/ - Stanford Education

Why do deep convolutional networks generalize so poorly to small image transformations?: https://arxiv.org/pdf/1805.12177.pdf - Aharon
Azulay, Yair Weiss May 2018

How to Resize, Pad Image to Square Shape and Keep Its Aspect Ratio With Python: https://jdhao.github.io/2017/11/06/resize-image-to-
square-with-padding/ - Jiedong Hao November 2017

Rotate images (correctly) with OpenCV and Python: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-
and-python/ - Adrian Rosebrock January 2017

Understanding regularization for image classification and machine learning: https://www.pyimagesearch.com/2016/09/19/understanding-
regularization-for-image-classification-and-machine-learning/ - - Adrian Rosebrock September 2016

## About the author(s)  
My name is Prem, and I am currently working as a freelance consultant specializing in SAP-ABAP and Android. I have a total of 12 years of experience and have just completed a course in Machine Learning. My primary focus is Image Classification using Keras and Tensorflow. The learning I garner is generally task oriented. My colleagues on this project are Satyajit Nair and Vivek V. Krishnan (https://github.com/vvkishere).
