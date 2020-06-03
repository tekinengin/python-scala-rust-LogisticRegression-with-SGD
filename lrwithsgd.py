
from random import seed
from random import randrange
from csv import reader
from math import exp
import argparse

# Load a CSV file
def load_csv(data):
    dataset = list()
    with open(data, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            temp = list()
            if not row:
                continue
            if len(row) == 1:
                dataset.append(float(row[0].strip()))
            else:
                for s in row:
                    temp.append(float(s.strip()))
                dataset.append(temp)
    return dataset

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def train_test_split(dataset, labels, ratio=0.3):
    dataset_train, dataset_test = list(dataset), list()
    label_train, label_test = list(labels), list()
    while len(dataset_test) < (len(dataset) * ratio):
        index = randrange(len(dataset_train))
        dataset_test.append(dataset_train.pop(index))
        label_test.append(label_train.pop(index))
    return dataset_train,label_train,dataset_test,label_test

# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = 0
    for i in range(len(row)):
        yhat += coefficients[i] * row[i]
        
    try:
        return 1.0 / (1.0 + exp(-yhat))
    except OverflowError:
        return 0
    
# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(x_train,y_train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(x_train[0]))]
    for _ in range(n_epoch):
        for i,row in enumerate(x_train):
            yhat = predict(row, coef)
            error = y_train[i] - yhat
            for i in range(len(row)):
                coef[i] = coef[i] + l_rate * error * yhat * (1.0 - yhat) * row[i]
            
    return coef

# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(x_train,y_train,x_test,y_test, l_rate, n_epoch):
    predictions = list()
    coef = coefficients_sgd(x_train,y_train, l_rate, n_epoch)
    for row in x_test:
        yhat = predict(row, coef)
        yhat = round(yhat)
        predictions.append(yhat)
    return accuracy_metric(y_test, predictions)


if __name__ == '__main__':  
    
    parser = argparse.ArgumentParser(description='Logistic Regression with SGD on Moon_Data')
    parser.add_argument('--n_size', type=int, help='int 1e3/1e4/1e5/1e6',default=int(1e3))
    parser.add_argument('--l_rate', type=float, help='learning rate',default=0.5)
    parser.add_argument('--n_epochs', type=int, help='number of epochs',default=5)
    args = parser.parse_args()
    seed(20)
    # load and prepare data
    data_file = 'data/moon_data_{}.csv'.format(args.n_size)
    label_file = 'data/moon_labels_{}.csv'.format(args.n_size)
    dataset = load_csv(data_file)
    labels = load_csv(label_file)
    x_train,y_train,x_test,y_test = train_test_split(dataset,labels)
    # evaluate algorithm
    scores = logistic_regression(x_train,y_train,x_test,y_test, args.l_rate, args.n_epochs)
    print('Scores: %s' % scores)
