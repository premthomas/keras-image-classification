# Grey-scale Image Classification using KERAS

## Disclaimer
This is a research project submitted for credit for a course that we are currently taking. The results seen here are subjective and should not be considered as final or accurate. 

## 1. Objective
During the history of computers (and computing), we have seen a significant rise in computing capabilities and a proportional drop in the cost of resources (such as CPUs, and memory). Note that during this period, wants and aspirations have also transcended from when we yearned for owning a computer one day, to the present wherein we have mobile devices (often a 100 times more powerful) in our very own pockets (or perhaps just strewn about our homes?)!

Using a similar thought process, we wish to submit that today's scenario - where more data leads to better learning, which in turn leads to accurate predictions - would soon change to a scenario where accuracy may be sacrificed for relatively quick training (learning) and deployment. In light of this, we plan on using only our laptops (see configuration below) and a fairly stable internet connection. We will deliberately be staying away from online cloud computing platforms, as students typically would not have the resources to pay for such platforms, nor sufficient time to utilize the free versions owing to their contraints on duration of usage. 

Thus, our objectives include:
  1. Classifying images from the Flickr27 dataset. Starting with three classes and gradually increasing until we breach our memory or computation resources.
  2. Keeping training time to a minimum. Minimum is subjective. Here, we are keeping the training time between 6-8 hours. Any model that takes longer would be difficult to use for prediction and testing of results at the end of the day.
  3. Using the least amount of resources possible. We are talking about limitations of the computer that is being used. Basis recommendations from leaders in the research space, the computers that we are using will be slightly higher that the minimum system requirements but a lot lower than the recommended systems. Please look at the section on System Configuration for more details.

Our target accuracy will be anything over 90%.

## 2. Learnings so far ...
On the offset, let me remind you that we have three months to complete this project. 

There are two kinds of Classifiers that exist. One is a Binary classifier (like Dog Vs. Cat) and the multi-class classifier. To understand if our algorithm works correctly, we will have to use 3 or more classes. We will start with three for our understanding and check to see how we cn increase the number of classes. The rules of Flickr27 state that each picture will contain only one of the 27 logos which have been categorised. Hence our code will evaluate the whole image at once and will categorise it as one of the three classes that we choose. We currently do not intend on training a background class which acts as a fail-safe. (should the logo detected not be in one of the three classes we have trained it for.

In terms of data, our understanding is the more we have, the more accurate our prediction will be. In keeping with our objective, we will be using the Flickr27 dataset. With regards to the algorithm that we intend on using, we will be testing CNNs (Convolutional Neural Networks). Our intention is to test smaller custom architectures and then move to larger ones with the understanding being that smaller networks would take lesser time to train. 

### How small is too small?
To answer this we know that 25 to 30 images is just too less to train any model. The loss function immediately takes a dive for the worse and the training platueas. Sources state that we will need at least a 1000 images per class to make make a valid impact on our training and test. Augmentation is the answer to this. We augment each image with a set of rules, thereby creating similar images with some noise or change to it. As far as possible, we will have to imitate the real-world scenarios and therefore understand that certain augmentations are just out of scope.

## 3. Software Requirements
In terms of software requirements, we will be using the following
  - Python with Keras and Tensorflow
  - GPU drivers and associated installations. Please refer to the link (https://www.tensorflow.org/install/install_windows) to check if and how to install the GPU
  - Most of our programming will happen on Jupyter notebooks rather than python programs as we will require to view output on a line-by-line execution level.
  - Higly suggested (but not mandatory) is installing Anaconda. This will help you create seperate environments in which you can execute you projects on. Should any of the libraries that we use be upgraded or changed, the failure would be contained within the environment and would not affect all the other developments that you have.

## 4. System configuration
Laptop with
  - Windows 10
  - Intel Core i5 - 7200U 2.5 GHz
  - NVIDIA GeForce 940MX with 2 GB dedicated VRAM
  - 8 GB DDR4 Memory
  - 1 TB HDD
 
This is close to the minimum requirements required to do a small scale image calssification project. Standard requirements would include
  - Intel Core i7
  - NVIDIA GTX series
  - 16 GB DDR4 Memory and 
  - 1 TB HDD
  
I would recommend that you look at Siraj's video that was posted on June 2018. Best Laptop for Machine Learning (https://www.youtube.com/watch?v=dtFZrFKMiPI). And yes, I would highly recommend other videos on Machine learning posted by Siraj. 
  
## 5. Basic code files
Here is a list of the code files that were used and their function.

## 6. Conclusion
Well, this went better than expected. With a 95% training, 95% test, and a 93% validation, this algorithm does show it is possible to create a highly accurate model with less data. 

What about testing with images taken from a camera? Would it work then? The answer is no. The results were very poor. No matter how well we controlled the method of taking an image for image classification, it almost always got it wrong. 

What if we put further constraints (like size) and used images from Google? While this did work better than images from real-life, it did not prove to be as accurate as that of the images that were present in Flickr27 and did not come close to the accuracies that were presented. The testing we did was small and it might be possible that the accuracies would have increased if we tested with a lot more images. 

## 7. What changes would I make?
  1. Find a way to compare images and get a score of the similarity between them. This way we remove duplicates from our train and test sets. Thus reducing the amout of time for training.
  2. Change the algorithm to use RGB images instead of Grey-scale images as most experts suggest that we will lose details that are important when converting the images from RGB to Grey-scale.
  3. Find a method of checking what is being detected in the image that is used for predtiction. This will help us understand the reasons why classification might go wrong. 
  4. Dectecting multiple logos in an image. For this alogorithm to have any meaningful revenue generation, our next steps should include methods of detecting (and classifying) multiple classes in one image and providing accuracy percentages for each of the detected classes.

## Citations, Credits, Sources, and References
Filckr27 - Y. Kalantidis, LG. Pueyo, M. Trevisiol, R. van Zwol, Y. Avrithis. Scalable Triangulation-based Logo Recognition. In Proceedings of ACM International Conference on Multimedia Retrieval (ICMR 2011), Trento, Italy, April 2011.

## About the author  
My name is Prem and I am currently working as a freelance consultant in SAP-ABAP and Android. I have a total of 14 years of experience and am currently attending a course in Machine Learning. My primary focus is on Image Classification using Keras and Tensorflow. The learning I do is generally task oriented. My colleagues of this project are Satyajit (The Doctor) and Vivek (The Visonary). While Satya is will be following a more methodical approach to understanding the technical aspects of the code, Vivek will be working on the business front of this application and visualisation of the data that is being collected.
