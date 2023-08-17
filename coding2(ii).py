import numpy as np
import math
from numpy import array
from PIL import Image
import matplotlib.pyplot as plt

def power_method(A, iterations):
    a, sigma = 0, 1
    x = np.random.normal(a, sigma, size=A.shape[1])
    B = A.T.dot(A)
    for i in range(iterations):
        new_x = B.dot(x)
        x = new_x
    left_singular_vector = x / np.linalg.norm(x)
    sigma = np.linalg.norm(A.dot(left_singular_vector))
    right_singular_vector = A.dot(left_singular_vector) / sigma
    return np.reshape(
        right_singular_vector, (A.shape[0], 1)), sigma, np.reshape(
        left_singular_vector, (A.shape[1], 1))


im_1 = Image.open(r"/Users/riwadesai/Documents/mathfordatascience/photo1.jpg")
img = array(im_1)
A = np.mean(img, axis=2)
grayscale_image = np.mean(img, axis=2)
rank = np.linalg.matrix_rank(A)
left_singular_matrix = np.zeros((A.shape[0], 1))
S = []
right_singular_matrix = np.zeros((A.shape[1], 1))

# Define the number of iterations
p = 0.001
q = 0.97
r = 2
iterations = int(math.log(
    4 * math.log(2 * A.shape[1] / p) / (q * p)) / (2 * r))

    # SVD using Power Method
for i in range(rank):
    right_singular_vector, sigma, left_singular_vector = power_method(A, iterations)
    left_singular_matrix = np.hstack((left_singular_matrix, right_singular_vector))
    S.append(sigma)
    right_singular_matrix = np.hstack((right_singular_matrix, left_singular_vector))
    A = A - right_singular_vector.dot(left_singular_vector.T).dot(sigma)

k = 30  # Choose the desired k (number of singular values to keep)
reconstructed_image = np.dot(left_singular_matrix[:, :k], np.dot(np.diag(S[:k]), right_singular_matrix.T[:k, :]))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(grayscale_image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title(f"Reconstructed Image (k = {k})")
plt.axis('off')

plt.tight_layout()
plt.show()
