# Classifying images of handwritten numbers


This program was created as part of the "Introduction to Data Science" course i took. all images are taken frome the MNIST database, and constructed from 28x28 grayscale pixels. 
It uses the The K-means algorithm, which is an unsupervised clustering iterative algorithm that partitions n observations into k clusters based on the nearest mean or centroid of a cluster, with the objective of minimizing the sum of squared distances between data points and their assigned cluster center. at the end of the K-means procces each image is classified as a digit 

The success rates were calculated using two different methods: 
1. initializing random centers
2. initializing the centers to be the mean of 10 reduced images that are picked from each label.

A clear difference can be seen.

The program is written in Python.

Installation:

Download the program files from the repository

Install Python 3.6 or higher

Install the required Python libraries: numpy and matplotlib

Command: pip install numpy matplotlib
