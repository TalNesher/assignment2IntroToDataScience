# Classifying images of handwritten numbers


This program was created as part of the "Introduction to Data Science" course I took. All images are taken from the MNIST database and constructed from 28x28 grayscale pixels.

It uses the K-means algorithm, which is an iterative, unsupervised clustering algorithm used to group data points into K clusters, with the objective of minimizing the sum of squared distances between data points and their assigned cluster center. It involves the following steps:

Initialization: Randomly choose K centroids **
Assignment: Calculate the distance between each data point and centroids, assigning each point to the nearest centroid.
Update: Recalculate the centroids based on the mean values of the data points in each cluster.
Iteration: Repeat the assignment and update steps until convergence or the maximum number of iterations is reached.
Result: Obtain K clusters, each represented by its centroid.

At the end of the K-means process, each cluster is classified as a digit, and the centroids are kept. Then, we assign each image from the test images to the closest centroid, meaning it was classified as the digit which that centroid represents.

** The success rates were calculated using two different methods:

Initializing random centers.
Initializing the centers to be the mean of 10 reduced images that are picked from each label.
A clear difference can be seen.

The program is written in Python.

Installation:

Download the program files from the repository; a folder with the datasets is also included.

Install Python 3.6 or higher.

Install the required Python libraries: numpy and matplotlib.

Command: pip install numpy matplotlib.

Make sure the root to the files is maching thire location on your machine.
