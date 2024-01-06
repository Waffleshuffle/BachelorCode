import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from annoy import AnnoyIndex



np.random.seed(42)

# Create a grid of values for x, y
x = np.linspace(-5, 5, 150)
y = np.linspace(-5, 5, 150)
x, y = np.meshgrid(x, y)

# Define the parameters for the bell curve
mu = [0, 0]
sigma = [[1, 0], [0, 1]]  # Covariance matrix

# Calculate the bell curve values
z = 0- np.exp(-0.5 * (np.square((x - mu[0]) / sigma[0][0]) + np.square((y - mu[1]) / sigma[1][1])))

# Generate random sample points on the surface of the bell curve
num_samples = 150
theta = 2 * np.pi * np.random.rand(num_samples)
radius = 4 * np.sqrt(np.random.rand(num_samples))  # Square root for uniform distribution

# Convert polar coordinates to Cartesian coordinates
rand_x = radius * np.cos(theta)
rand_y = radius * np.sin(theta)
random_samples = np.column_stack([rand_x, rand_y])
rand_z = 0 - np.exp(-0.5 * (np.square(random_samples[:, 0] / sigma[0][0]) + np.square(random_samples[:, 1] / sigma[1][1])))


K_VALUE = 1

def calculateError(base, sim):
    error = 0
    for i in range(len(x)):
        for j in range(len(y)):
            error = error + (base[i,j] - sim[i,j])**2
    return error / (len(x)*len(y))

def getCurForVec(vector):
    knn = index.get_nns_by_vector(vector, K_VALUE, -1, True)
    #knn is sorted by distance

    # checking if the closest neighbour has distance 0
    if (knn[1][0] == 0):
        return target[knn[0][0]] # if so we can just return the actual value

    distances = np.array(knn[1])

    totalDist = np.sum(distances)
    weights = totalDist/distances
    sumWeights = np.sum(weights)
    weights = weights/sumWeights

    out = 0
    for i in range(K_VALUE):
        out = out + weights[i]*target[knn[0][i]]

    return out

index = AnnoyIndex(2,'euclidean')
target = []
for i in range(num_samples):
    add = np.array([rand_x[i], rand_y[i]])
    target.append(rand_z[i])
    index.add_item(i, add)
target = np.array(target)
print("knn-Sim: Index filled")
index.build(3)
print("knn-Sim: Index build")




k = 1
errors = []
while k <= 32:
    K_VALUE = k
    simZ = np.zeros_like(x)
    for i in range(len(x)):
        for j in range(len(y)):
            simZ[i, j] = getCurForVec([x[i, j], y[i, j]])
    errors.append(calculateError(z,simZ))
    k = k+1

print(errors)

