import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def importdata():
    #This dataset consists of 101 animals from a zoo. There are 16 variables with various traits to describe the animals.
    #The 7 Class Types are: Mammal, Bird, Reptile, Fish, Amphibian, Bug and Invertebrate.
    #The purpose for this dataset is to be able to predict the classification of the animals, based upon the variables. 
    #It is the perfect dataset for those who are new to learning Machine Learning.
    data = pd.read_csv("zoo.csv")
    data = data.drop('animal_name', axis=1)
    return data


def splitdata(data):
    X = data.iloc[:, 0:-1].values
    Y = data.iloc[:, -1].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
    return X, Y, X_train, X_test, Y_train, Y_test


def train_using_gini(X, Y):
    clf = DecisionTreeClassifier()
    clf.fit(X, Y)
    return clf


def train_using_entropy(X, Y):
    clf_entropy = DecisionTreeClassifier(criterion="entropy")
    clf_entropy.fit(X, Y)
    return clf_entropy


def prediction(X_test, clf_obj):
    Y_pred = clf_obj.predict(X_test)
    print('Prediction Result')
    print(Y_pred)
    return Y_pred


def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)

    print("Report : ",
          classification_report(y_test, y_pred))


def main():
    data = importdata()
    X, Y, X_train, X_test, Y_train, Y_test = splitdata(data)
    clf_gini = train_using_gini(X_train, Y_train)
    clf_entropy = train_using_entropy(X_train, Y_train)

    print('Result using gini')

    Y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(Y_test, Y_pred_gini)

    print('Result using Entropy')

    Y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(Y_test, Y_pred_entropy)


if __name__ == '__main__':
    main()

