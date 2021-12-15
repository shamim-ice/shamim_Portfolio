import pandas as pd
import numpy as np
import operator

data = pd.read_csv('IRIS.csv')
data = data.dropna()


def train_test_split(data):
    training_data = data.iloc[:100].reset_index(drop=True)
    testing_data = data.iloc[100:].reset_index(drop=True)
    return training_data, testing_data


def euclid(test_data, train_data, length):
    dist = 0
    for x in range(length):
        dist += (train_data[x] - test_data[x]) * (train_data[x] - test_data[x])
    return np.sqrt(dist)


def K_NN(training_data, testing_data, k):
    distance = {}
    length = 4

    for x in range(len(training_data)):
        dist = euclid(testing_data, training_data.iloc[x], length)
        distance[x] = dist

    sorted_d = sorted(distance.items(), key=operator.itemgetter(1))
    neighbors = []

    for i in range(k):
        neighbors.append(sorted_d[i][0])

    res = pd.DataFrame(columns=['res'])
    for i in range(len(neighbors)):
        response = training_data.iloc[neighbors[i]][-1]
        res.loc[i, 'res'] = response

    elements, counts = np.unique(res['res'], return_counts=True)
    mx_index = np.argmax(counts)

    return elements[mx_index]


training_data = train_test_split(data)[0]
testing_data = train_test_split(data)[1]

k = np.sqrt(len(data))
if k % 2 == 0:
    k += 1

predicted = pd.DataFrame(columns=['predict'])

for x in range(len(testing_data)):
    result = K_NN(training_data, testing_data.iloc[x], int(k))
    predicted.loc[x, 'predict'] = result

print('The prediction accuracy is : ', (np.sum(predicted['predict'] == testing_data['class']) / len(testing_data)) * 100)
