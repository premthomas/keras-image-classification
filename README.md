# Grey-scale Image Classification using KERAS

## Disclaimer
This is a research project submitted for credit for a course that we just completed. The results seen here are subjective and should not be considered as final or accurate. 

## 1. Objective
When considering the history of computers (and computing), we have seen a significant rise in computing capabilities and a proportional drop in the cost of resources (such as CPUs, and memory). Note that during this period, wants and aspirations have also transcended - from the times we yearned for owning a computer one day, to the present wherein we have mobile devices (often a 100 times more powerful) in our very own pockets (or perhaps just strewn about our homes?)!

Using a similar thought process, we wish to submit that today's scenario - where more data leads to better learning, which in turn leads to accurate predictions - would soon change to a scenario where accuracy may be sacrificed for relatively quick training (learning) and deployment. In light of this, we plan on using only our laptops (see configuration below) and a fairly stable internet connection. We will deliberately be staying away from online cloud computing platforms, as students typically would not have the resources to pay for such platforms, nor sufficient time to utilize the free versions owing to their contraints on duration of usage. 

Thus, our objectives include:
  1. Classifying images from the Flickr27 dataset: Starting with three classes and gradually increasing until we breach our memory or computation resources.
  2. Keeping training time to a minimum. Minimum is subjective: Here, we are keeping the training time between 6-8 hours. Any model that takes longer would be difficult to use for prediction and testing of results at the end of the day.
  3. Using the least amount of resources possible: Here, we are referencing the limitations of the computer that is being used. Basis recommendations from leaders in the research space, the computers that we are using will be slightly higher that the minimum system requirements, but a lot lower than the recommended systems. Please refer the section on System Configuration for more details.

Our target (test) accuracy will be anything over 90%.

## 2. Learnings so far ...
At the outset, let me remind you that we had three months to complete this project. 

There are two kinds of Classifiers that exist. The Binary classifier (like Dog vs. Cat) and the Multi-class classifier. To understand if our algorithm works correctly, we will have to use 3 or more classes. We start with three for our understanding, and check to see how we can subsequently increase the number of classes. The rules of Flickr27 state that each picture will contain only one of the 27 logos which have been categorised. Hence our code will evaluate the whole image at once and will categorise it as one of the three chosen classes. We currently do not intend on training a background class which acts as a fail-safe (should the logo detected not be in one of the three classes we have trained it for). Thinking about it in the most basic way, 3 classes would mean that if the model was to take a random guess, the prediction would be right 33.3% of the time. Any increase from here suggests that the model is learning and as the number continually gets higher, it would be difficult to improve on it. 

In terms of data, our understanding is that the more we have, the more accurate our prediction will be. In keeping with our objectives, we will be using the Flickr27 dataset. Regarding the algorithm that we intend on using, we will be testing CNNs (Convolutional Neural Networks). Our intention is to test smaller custom architectures and then move to larger ones with the understanding being that smaller networks would take lesser time to train.

Getting the GPU to work was one of our bigger hurdles. Based on the augmentation and the number of classes, we ran training scenarios for 20 epochs which took us over 12 hours. Only after going through a large number of blog posts, did we finally get our GPU's working and it did give us a minimum of 10 times the improvement. However, one disadvantage is that the memory limitation in the GPU did not allow us to use more classes. 

### How small is too small?
25 to 30 images is just too small to train any model. The loss function immediately takes a nose dive and the training platueas. Sources state that we will need at least 1000 images per class to allow for a valid impact on our training and tests. Augmentation is the answer to this. We augment each image with a set of rules, thereby creating similar images with some noise or changes to it. As far as possible, we will have to imitate real-world scenarios and therefore understand that certain augmentations are just out of scope.

## 3. Software requirements
In terms of software requirements, we will be using the following
  - Python with Keras and Tensorflow
  - GPU drivers and associated installations. Please refer to the link (https://www.tensorflow.org/install/install_windows) to check if and how to install the GPU
  - Most of our programming will happen on Jupyter notebooks rather than python programs, as we will require to view output on a line-by-line execution level
  - Highly suggested (but not mandatory) is installing Anaconda. This will help you create seperate environments in which you can execute your projects. Should any of the libraries that we use be upgraded or changed, the failure would be contained within the environment and would not affect all the other developments that you have

## 4. More about the dataset
Flick27 is a collection of 27 classes (such as Apple, Google, McDonalds) with each class containing about 35 images. The dataset is already broken up into "train" (30 images) and "test" (5 images) sets. When training any model, we need to have a train and validation set, we therefore broke the train set into two sets: a train (24 images) and a validation (6 images). It is best that you put all the image files into subfolders whose names represent the class to which it belongs. The test set should not be used until you have acceptable training and validation accuracies. 

## 5. System configuration
Laptop with:
  - Windows 10
  - Intel Core i5 - 7200U 2.5 GHz
  - NVIDIA GeForce 940MX with 2 GB dedicated VRAM
  - 8 GB DDR4 Memory
  - 1 TB HDD
 
This is close to the minimum requirements necessary to run a small scale image calssification project. Standard requirements would include:
  - Intel Core i7
  - NVIDIA GTX 970
  - 16 GB DDR4 Memory and 
  - 1 TB HDD
  
I would recommend that you look at Siraj's video that was posted on June 2018. Best Laptop for Machine Learning (https://www.youtube.com/watch?v=dtFZrFKMiPI). And yes, I would highly recommend other videos on Machine Learning posted by Siraj. 
  
## 6. Libraries used
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

## 7. Basic code files
Here is a list of the code files that were used and their functions:
  - CreateModel.ipynb: Your first step is to create a model. There are two ways of creating models. You could import a model programmed in Keras directly (read this link for information on available models https://keras.io/applications/) or you could create your own model. In this case, we will be creating our own model using InceptionV3 as the base. The reason in doing so is that most models work with RGB images only and not with Grey-scale. There are a few variables that you will have to change:
    - Number of channels the image has: 1 represents a Grey-scale image, 3 represents a RGB (or HSV) image
    - Number of classes: This is important as this will represent your final output layer. In this example, the value is set to 3
    
  - TrainModel.ipynb: The next step is to train your model. This step could be the most time consuming process. Remember that this will depend on the system and its configuration that is available. In this example, we ran 100 epochs, each of which took approximately 200 seconds. Notice that in the 67th epoch, we have a training accuracy of 99.97% and a validation accuracy of 98.33%. This does represent an overfitting problem but only very slightly
  
  - TestModel.ipynb: Finally, we use the trained model (with weights) and predicted classes for the images that we have in our validation set. The results are not as good as we expected. It was 13 correct predictions out of the 15 available, and this translated to 86.6% accuracy.  However, this can be due to the number of images available for testing being very low.
  
## 8. Conclusion
With a 99.97% training, 98.33% validation, and a 86.66% test, this algorithm does show it is possible to create a highly accurate model with less data. 

What about testing with images taken from a camera? Would it work then? The answer is no. The results were very poor. No matter how well we controlled the method of taking an image for image classification, it almost always got it wrong. The score in this case dropped at a little less than 60%. 

What if we put further constraints (like size) and used images from Google? While this did work better than images from real-life, it did not prove to be as accurate as that of the images that were present in Flickr27 and did not come close to the accuracies that were presented. To reiterate, the testing we conducted was small and it might be possible that the accuracies would have increased if we tested with a lot more images. 

## 9. What changes would I make?
  1. Find a way to compare images and get a score of the similarity between them. This way we remove duplicates from our train and test sets, thus reducing the training time
  2. Change the algorithm to use RGB images instead of Grey-scale images as most experts suggest that we will lose details that are important when converting the images from RGB to Grey-scale
  3. Find a method of checking what is being detected in the image that is used for prediction. This will help us understand the reasons behind why the classification could possible go wrong
  4. Dectecting multiple logos in an image. For this alogorithm to have any sort of meaningful revenue generation, our next steps should include methods of detecting (and classifying) multiple classes in one image and providing accuracy percentages for each of the detected classes

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
My name is Prem, and I am currently working as a freelance consultant specializing in SAP-ABAP and Android. I have a total of 14 years of experience and have just concluded a course in Machine Learning. My primary focus is on Image Classification using Keras and Tensorflow. The learning I garner is generally task oriented. My colleagues on this project are Satyajit Nair and Vivek V Krishnan.
