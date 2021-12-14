import pandas as pd
import numpy as np
from pprint import pprint


#This dataset consists of 101 animals from a zoo. There are 16 variables with various traits to describe the animals.
#The 7 Class Types are: Mammal, Bird, Reptile, Fish, Amphibian, Bug and Invertebrate.
#The purpose for this dataset is to be able to predict the classification of the animals, based upon the variables. 
#It is the perfect dataset for those who are new to learning Machine Learning.
dataset = pd.read_csv("zoo.csv")
dataset = dataset.drop('animal_name', axis=1)


def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)

    entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * (np.log2(counts[i] / np.sum(counts))) for i in range(len(elements))])
    return entropy


def infogain(data, split_attribute_name, target_name='class'):
    total_entropy = entropy(data[target_name])

    vals, counts = np.unique(data[split_attribute_name], return_counts=True)

    weighted_entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])
    IG = total_entropy - weighted_entropy
    return IG


def ID3(data, originaldata, features, target_attribute_name='class', parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name][
                             np.argmax(np.unique(originaldata[target_attribute_name][1]))])

    elif len(features) == 0:
        return parent_node_class

    else:
        parent_node_class = np.unique(originaldata[target_attribute_name][
                                          np.argmax(np.unique(originaldata[target_attribute_name][1]))])

        item_values = [infogain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature: {}}

        features = [i for i in features if i != best_feature]

        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            sub_tree = ID3(sub_data, training_data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = sub_tree

        return tree


def train_test_split(dataset):
    training_data = dataset.iloc[:80].reset_index(drop=True)
    testing_data = dataset.iloc[80:].reset_index(drop=True)
    return training_data, testing_data


def predict(query, tree, default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default

            result = tree[key][query[key]]

            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result


def test(data, tree):
    queries = data.iloc[:, :-1].to_dict(orient='records')
    predicted = pd.DataFrame(columns=['predicted'])

    for i in range(len(data)):
        predicted.loc[i, 'predicted'] = predict(queries[i], tree, 1.0)

    print('The prediction accuracy is : ', (np.sum(predicted['predicted'] == data['class']) / len(data)) * 100, '%')


training_data = train_test_split(dataset)[0]
testing_data = train_test_split(dataset)[1]
tree = ID3(training_data, training_data, training_data.columns[:-1])
pprint(tree)
test(testing_data, tree)
