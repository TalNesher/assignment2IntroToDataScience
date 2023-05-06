import numpy as np
import struct
from array import array
import os
import matplotlib.pyplot as plt

class MnistDataloader(object):
# This class loads the MNIST images and labels from their respective files. 

    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        # Read labels
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError("Magic number mismatch, expected 2049, got {}".format(magic))
            labels = array("B", file.read())

        # Read images
        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError("Magic number mismatch, expected 2051, got {}".format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        # Converting the lists into NumPy arrays
        images = np.array(images)
        labels = np.array(labels)

        return images, labels

    def load_data(self):
        images_train, labels_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        images_test, labels_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (images_train, labels_train),(images_test, labels_test)


#----------------------------------------------main---------------------------------------------

# Set file paths based on added MNIST Datasets
MNIST_data_dir = "MNIST data"
training_images_filepath = os.path.join(os.getcwd(), MNIST_data_dir, "train-images.idx3-ubyte")
training_labels_filepath = os.path.join(os.getcwd(), MNIST_data_dir, "train-labels.idx1-ubyte")
test_images_filepath = os.path.join(os.getcwd(), MNIST_data_dir, "t10k-images.idx3-ubyte")
test_labels_filepath = os.path.join(os.getcwd(), MNIST_data_dir, "t10k-labels.idx1-ubyte")


# Load MINST dataset
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(images_train, labels_train), (images_test, labels_test) = mnist_dataloader.load_data()


# Reduce all images to between 0.5 to -0.5
images_train = (images_train / 255.0) - 0.5
images_test = (images_test / 255.0) - 0.5
m = images_train.shape[0]

# Flattening the images
X = (images_train.reshape(images_train.shape[0], -1)).T

# Compute the covariance matrix
cov_matrix = np.dot(X, X.T) / m

# Compute the eigendecomposition of the covariance matrix
U, S, Ut = np.linalg.svd(cov_matrix)
S_squared = np.sqrt(S)

# Plot the singular values, can be commented-out if not needed
# plt.plot(S_squared)
# plt.title('Square root of singular values')
# plt.xlabel('Index')
# plt.ylabel('Singular value')
# plt.show()

# Select the first p columns of U as Up
p = 40
Up = U[:, :p]

# Compute the reduced-sized vectors for each image. now each coulm in reduced_images_train represent a picture with 40 elements instead of 784
reduced_images_train = np.dot(Up.T, X)

# Reconstruct an image from the reduced-sized vector
i = 1 # Choose an image index between 0 and 59,999
image_reconstructed = np.dot(Up, reduced_images_train[:, i])

# Reshape the reconstructed image back to a 2D array
image_reconstructed = image_reconstructed.reshape(28, 28)

# Plot the original and reconstructed images, can be commented-out if not needed
# plt.subplot(1, 2, 1)
# plt.imshow(images_train[i], cmap="gray")
# plt.title("Original image")
# plt.subplot(1, 2, 2)
# plt.imshow(image_reconstructed, cmap="gray")
# plt.title("Reconstructed image")
# plt.suptitle("Comparison of Original and Reconstructed Images", fontsize=16)
# plt.show()


def kmeans(images_matrix, num_of_clusters, centers_kmeans ,max_iter=10):
    #Performs the K-means clustering algorithm on a matrix of images.
    for i in range (max_iter):
        assigned_centers = assign_to_closest_center(images_matrix, centers_kmeans)
        centers_kmeans = recompute_centers(images_matrix, assigned_centers, num_of_clusters)
    
    # Plot the images (as points) and centers (as 'x'), each cluster's points in different color
    # fig = plt.figure(figsize=(10, 8))
    # plt.scatter(images_matrix[:, 0], images_matrix[:, 1], s=5, c = assigned_centers, cmap = "viridis")
    # plt.scatter(centers_kmeans[:, 0], centers_kmeans[:, 1], c = "red", marker = "x")
    # plt.xlabel("First Dimension")
    # plt.ylabel("Second Dimension")
    # plt.title("K-Means Clustering")
    # plt.show()

    centers_after_kmeans = centers_kmeans
    return assigned_centers, centers_after_kmeans


def assign_to_closest_center(images_matrix, centers_kmeans):
    # Assigns each point(image) in the images_matrix to the closest center in centers_kmeans.
    assigned_centers = np.argmin(np.linalg.norm(images_matrix[:, np.newaxis,:] - centers_kmeans, axis=2), axis=1)
    return assigned_centers


def recompute_centers(images_matrix, assigned_centers, num_of_clusters):
    # Recomputes the centers of each cluster based on the assigned images.
    centers_kmeans = np.array([images_matrix[assigned_centers == i].mean(axis=0) if np.sum(assigned_centers == i) != 0 else np.random.uniform(low = -0.5, high = 0.5, size = images_matrix.shape[1]) for i in range(num_of_clusters)])
    return centers_kmeans


print("first run of kmeans:")
# Run kmean with p = 40 and randomize centers
num_of_clusters = 10

# Define each row to be image, and each coulmn to be element
reduced_images_train = reduced_images_train.T

# Initialize the centroids randomly
centroids = np.random.uniform(low = -0.5, high = 0.5, size = (num_of_clusters, reduced_images_train.shape[1]))

assigned_centers_to_train_images, centers_after_kmeans = kmeans(reduced_images_train, num_of_clusters, centroids)
assigned_centers_to_train_images = np.array(assigned_centers_to_train_images)


# Assign a digit to a cluster using the most common label in that cluster
def give_centers_lebles(assigned_centers, labels_train):
    sums_of_images_per_center = np.zeros((num_of_clusters, num_of_clusters))
    centers_to_labels = np.zeros(num_of_clusters)
    for i in range (assigned_centers.shape[0]):
        image_real_lable = labels_train[i]
        image_assigned_center = assigned_centers[i]
        sums_of_images_per_center[image_assigned_center, image_real_lable] += 1

    for j in range (num_of_clusters):
        approximate_center = np.argmax(sums_of_images_per_center[j])
        centers_to_labels[j] = approximate_center
    return centers_to_labels


centers_to_labels = give_centers_lebles(assigned_centers_to_train_images, labels_train)


# Now we prepare the test images to be classified
m_test = images_train.shape[0]
X_test = (images_test.reshape(images_test.shape[0], -1)).T

# Compute the covariance matrix
cov_matrix_test = np.dot(X_test, X_test.T) / m_test

# Compute the eigendecomposition of the covariance matrix
U_test, S_test, Ut_test = np.linalg.svd(cov_matrix_test)

# Select the first p columns of U as Up
p = 40
Up_test = U_test[:, :p]
# Compute the reduced-sized vectors for each image. now each coulm in reduced_images_test represent a picture with 40 elements instead of 784
reduced_images_test = np.dot(Up_test.T, X_test)
reduced_images_test = reduced_images_test.T

assigned_centers_test = assign_to_closest_center(reduced_images_test, centers_after_kmeans)


def caculate_succses(centers_to_labels, assigned_centers, labels, reduced_images):
    # Calculates the success rate of the K-means clustering algorithm.
    succsus = 0
    for i in range(reduced_images.shape[0]):
        kmeans_center_result = centers_to_labels[assigned_centers[i]]
        if labels[i] == kmeans_center_result:
            succsus += 1
    return (succsus / reduced_images.shape[0]) * 100


succses_presents = caculate_succses(centers_to_labels, assigned_centers_test, labels_test, reduced_images_test)
print("the succses rate is", succses_presents, "precents")


# Running for three iterations to check if differant sets of random centroids yiled differant succses rates
print("when chosing random centroids:")
for i in range(3):
    centroids = np.random.uniform(low = -0.5, high = 0.5, size = (num_of_clusters, reduced_images_train.shape[1]))
    assigned_centers_to_train_images, centers_after_kmeans = kmeans(reduced_images_train, num_of_clusters, centroids)
    assigned_centers_to_train_images = np.array(assigned_centers_to_train_images)
    centers_to_labels = give_centers_lebles(assigned_centers_to_train_images, labels_train)
    assigned_centers_to_test_images = assign_to_closest_center(reduced_images_test, centers_after_kmeans)
    succses_presents = caculate_succses(centers_to_labels, assigned_centers_to_test_images, labels_test, reduced_images_test)
    print("iteration", i, "has a succses rate of: ", succses_presents, "precents")


# Chosinig each starting centroid to be the mean of 10 images that are all labeld as the same number, for each number between 0 to 9
centroids = np.zeros((10, reduced_images_train.shape[1]))
for i in range(10):
        counter = 0
        index = 0
        while counter < 10:
            if(labels_train[index] == i):
                for p in range(reduced_images_train.shape[1]):
                    centroids[i, p] += reduced_images_train[index, p]
                counter += 1
            index += 1
        centroids[i] = centroids[i] / 10

assigned_centers_to_train_images, centers_after_kmeans = kmeans(reduced_images_train, num_of_clusters, centroids)
assigned_centers_to_train_images = np.array(assigned_centers_to_train_images)
centers_to_labels = give_centers_lebles(assigned_centers_to_train_images, labels_train)
assigned_centers_to_test_images = assign_to_closest_center(reduced_images_test, centers_after_kmeans)
succses_presents = caculate_succses(centers_to_labels, assigned_centers_to_test_images, labels_test, reduced_images_test)
print("when computing the centers we get ", succses_presents, "precent sucsses rate")
