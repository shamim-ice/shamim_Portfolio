import pandas as pd
import numpy as np

dataset = pd.read_csv("pima-indians-diabetes.csv")


def train_test_split(dataset):
    training_data = dataset.iloc[:500].reset_index(drop=True)
    testing_data = dataset.iloc[500:].reset_index(drop=True)
    return training_data, testing_data


training_data = train_test_split(dataset)[0]
testing_data = train_test_split(dataset)[1]

data_means = training_data.groupby("class").mean()
data_var = training_data.groupby("class").var()

elements, counts = np.unique(training_data['class'], return_counts=True)
P_class = [counts[i] / np.sum(counts) for i in range(len(elements))]


def P_X_given_Y(x, mean_y, var_y):
    p = 1 / (np.sqrt(2 * np.pi * var_y)) * np.exp((-(x - mean_y) ** 2) / (2 * var_y))
    return p


def test(data, elements, P_class, data_means, data_var):
    features = data.columns[:-1]
    predicted = pd.DataFrame(columns=['predicted'])

    for it in range(len(data)):
        c = 0
        m = 0
        for i in range(len(elements)):
            ind = elements[i]
            pp = P_class[i]
            for feature in features:
                pp = pp * P_X_given_Y(data[feature][it], data_means[feature][data_means.index == ind].values[0],
                                      data_var[feature][data_var.index == ind].values[0])
            if pp >= c:
                c = pp
                m = ind

        predicted.loc[it, 'predicted'] = m

    print('The prediction accuracy is : ', (np.sum(predicted['predicted'] == data['class']) / len(data)) * 100, '%')


test(testing_data, elements, P_class, data_means, data_var)
