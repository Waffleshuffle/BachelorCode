import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from annoy import AnnoyIndex

K_VALUE = 15

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

index = AnnoyIndex(2,'euclidean')
target = []
for i in range(num_samples):
    add = np.array([rand_x[i], rand_y[i]])
    target.append(rand_z[i])
    index.add_item(i, add)
target = np.array(target)
print("knn-Sim: Index filled")
index.build(10)
print("knn-Sim: Index build")



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


simZ = np.zeros_like(x)
for i in range(len(x)):
    for j in range(len(y)):
        simZ[i,j] = getCurForVec([x[i, j], y[i, j]])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the wireframe

ax.plot_surface(x, y, simZ, rstride=5, cstride=5, color='royalblue', edgecolors='b', linewidth=0.5, alpha=0.3)
ax.contourf(x, y, z, zdir='z', offset=-1.1, cmap='viridis', alpha=0.7)
# Plot the random sample points on the surface
ax.scatter(rand_x, rand_y,rand_z, c='black', marker='o', alpha = 1)
ax.scatter(random_samples[:, 0], random_samples[:, 1], -1.1, c='black', marker='o', alpha = 0.05)
# Show the plot
plt.savefig("Gaus_k15.png", dpi=300, bbox_inches='tight')
plt.show()


