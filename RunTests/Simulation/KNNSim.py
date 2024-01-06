
from annoy import AnnoyIndex
import numpy as np
import pickle
import random

K_VALUE = 150

file_path_states = 'Simulation/E0F1.pkl'
file_path_names = 'Simulation/E0F1Names.pkl'


#Loading the data
try:
    states = pickle.load(open(file_path_states, 'rb'))
    names = pickle.load(open(file_path_names, 'rb'))
except FileNotFoundError:
    print("Error: The files could not be found")
    exit(-1)

print(f"knn-Sim: Data loaded from {file_path_states}")


#knn training
index = AnnoyIndex(len(states[0][0]),'euclidean')
target = []
for i, vector in enumerate(states):
    add = np.array(vector[0])
    target.append(vector[1])
    index.add_item(i, add)
target = np.array(target)
print("knn-Sim: Index filled")
index.build(10)
print("knn-Sim: Index build")
print("knn-Sim: READY")

def setK(newK):
    K_VALUE = newK

def randK():
    K_VALUE = random.randint(5,100)

def get(vector):
    knn = index.get_nns_by_vector(vector, K_VALUE, -1, True)
    #knn is sorted by distance

    cosestDist = knn[1][0]

    # checking if the closest neighbour has distance 0
    if (cosestDist == 0):
        return target[knn[0][0]] # if so we can just return the actual value

    distances = np.array(knn[1])

    #implements our weighing strategy
    totalDist = np.sum(distances)
    weights = totalDist/distances
    sumWeights = np.sum(weights)
    weights = weights/sumWeights

    out = 0
    for i in range(K_VALUE):
        out = out + weights[i]*target[knn[0][i]]

    # now lets add our punishment for distance
    out = min(out + (cosestDist *10)**2,0)

    return out
