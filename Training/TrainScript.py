import tensorflow as tf
import numpy as np
import random
import math
import pickle
import Simulation.KNNSim as sim

print("Load Indexes")
startIndexes = pickle.load(open('Simulation/E0F1_index_transmision.pkl', 'rb'))
usablePVindextable = pickle.load(open('Simulation/E0F1_index_I1.pkl', 'rb'))
pvNames = pickle.load(open('Simulation/E0F1Names.pkl', 'rb'))

print("Build Model")
input_dim =  len(usablePVindextable)*2 + 1 # permissionbools for PVs (dependand on subsection),current PV values , current current value
output_dim = len(usablePVindextable)
input_dim
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True, stateful=True, batch_input_shape=(1,None,input_dim)),
    tf.keras.layers.LSTM(32, return_sequences=True, stateful=True),
    tf.keras.layers.LSTM(64, return_sequences=False, stateful=True),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(output_dim, activation='linear')
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.build()
model.compile()
model.summary()

def generateRandomBoolArr(length, numOfOnes):
    result = np.zeros(length, dtype=int)
    indices = np.random.choice(length, numOfOnes, replace=False)
    result[indices] = 1
    return result

def modelOutToPosition(modelOut, allowedPVs, oldPosition):
    for i, change in enumerate(modelOut):
        if allowedPVs[i]:
            oldPosition[usablePVindextable[i]] += change
        modelOut[i] = oldPosition[usablePVindextable[i]]
    return [oldPosition, modelOut]

def biasedRandomNumber(max_value):
    prob = np.arange(1, max_value + 1)
    # Normalize probabilities to sum to 1
    prob = prob / np.sum(prob)
    result = max_value + 1 - np.random.choice(np.arange(1, max_value + 1), p=prob)
    return result

#40104 is the index for the global optimum
optimum = list(sim.states[40104][0])
baseOptimum = []

for i in usablePVindextable:
    baseOptimum.append(optimum[i])

var_I1 = [0.4656249999999993,0.17000000000000035,0.6524999999999999,0.14566250000000042,0.34001250000000066,0.19999999999999996,0.14499999999999957,0.13628749999999912,2.239959716796875,1.40057373046875,2.3402099609375,1.9005126953125]

def get_StartingPossition():
    sectionPos = []
    for i, variance in enumerate(var_I1):
        newValue = baseOptimum[i] + (random.random()-0.5) * variance
        optimum[usablePVindextable[i]] = newValue
        sectionPos.append(newValue)
    return [np.array(optimum), np.array(sectionPos)]


# Training loop
runsPerepisode = 100
episodes = 30000

bestLoss = 0
bestDec = 0

num_steps = 4.9
print("Starting Traning...")
for episode in range(episodes):

    num_steps = min(num_steps + 0.01, 50)
    sumList = [0, 0, 0]
    for run in range(runsPerepisode):
        # This is one run
        sim.randK()

        # Reset the LSTM states
        model.layers[0].reset_states()
        model.layers[1].reset_states()
        model.layers[2].reset_states()

        # Choose a starting position
        position, sectionPos = get_StartingPossition()

        # Choose which PVs we can manipulate in this run
        allowedPVs = generateRandomBoolArr(output_dim, np.random.choice(np.arange(1, 13)))

        # Init loss
        y = sim.get(position)
        minY = y
        startY = y

        states = []
        returns = []
        action = []
        fail = 0
        with tf.GradientTape() as tape:
            # Forward pass
            for step in range(math.floor(num_steps)):

                modelIn = np.reshape(np.concatenate((allowedPVs, sectionPos, [y]), axis=0), (1, 1, input_dim))
                modelOut = model(modelIn)
                action.append(modelOut)
                # We now have our next step
                position, sectionPos = modelOutToPosition(modelOut.numpy()[0], allowedPVs, position)
                # We can calculate the loss right here
                y = sim.get(position)
                returns.append(y - startY)

                if y < minY:
                    minY = y
            # We made a run Now to calculate a gradient
            bestImp = returns[-1]
            i = -1
            while i >= -len(returns):
                if returns[i] < bestImp:
                    bestImp = returns[i]
                returns[i] = bestImp
                i -= 1

            result = [(action * allowedPVs / tf.reduce_sum(abs(action * allowedPVs))) * returns for action, returns
                      in zip(action, returns)]
            loss = tf.reduce_sum(result)

        # Backward pass
        gradients = tape.gradient(tf.constant(loss), model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        sumList[0] += loss
        sumList[1] += minY - startY
        if startY != minY:
            sumList[2] += 1

    # Print episode information
    print(
        f'episode {episode+1}/{episodes}, AVG Loss: {sumList[0]/runsPerepisode: .3f}, AVG improvment: {sumList[1]/runsPerepisode: .3f}, Acc: {(sumList[2]/runsPerepisode)*100: .1f}%')

    if sumList[0] / runsPerepisode < bestLoss:
        bestLoss = sumList[0] / runsPerepisode
        print(f'------------------------ New best Loss with {bestLoss: .3f} ------------------------')
        model.save('BestLoss.keras')

    if (sumList[1] / runsPerepisode) * (sumList[2] / runsPerepisode) < bestDec:
        bestDec = (sumList[1] / runsPerepisode) * (sumList[2] / runsPerepisode)
        print(f'------------------------ New best Dec with {bestDec: .3f} ------------------------')
        model.save('BestDec.keras')

    if episode % 100 == 0:
        name = f'ModelE{episode}_L{sumList[0]/runsPerepisode: .2f}_I{sumList[1]/runsPerepisode: .2f}_P{sumList[2]/runsPerepisode: .2f}'
        print("Modelsnapshot saved as:" + name)
        model.save(name + '.keras')
