#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 20:47:10 2019

@author: sethconklin
"""

from sklearn.model_selection import train_test_split
from math import exp
import pandas as pd
import numpy as np
import random

class Node:
    def __init__(self, value=0, weights=[], error=0):
        self.value = value
        self.weights = weights
        self.error = error

class NeuralNetClassifier:
    def __init__(self, hidden_numbers = [3, 3], learning_rate = 0.1):
        self.data = None
        self.target = None
        self.output_nodes = []
        self.hidden_nodes = []
        self.targets = None
        self.hidden_numbers = hidden_numbers
        self.learning_rate = learning_rate
        
    def create_nodes(self, input_number, output_number):
        previous_nodes = input_number + 1
        current_nodes = []
        for value in self.hidden_numbers:
            for x in range(value):
                current_nodes.append(Node(0,[]))
                for y in range(previous_nodes):
                    current_nodes[x].weights.append(random.uniform(-.9,.9))
            self.hidden_nodes.append(list(current_nodes))
            current_nodes = []
            previous_nodes = value + 1;
            
        for x in range(output_number):
            self.output_nodes.append(Node(0,[]))
            for y in range(previous_nodes):
                self.output_nodes[x].weights.append(random.uniform(-.9,.9))
                                
    def fit(self, data, targets, iterations=40):
        self.targets = list(set(targets))
        output_number = len(self.targets)
        input_number = len(data[0])
        self.create_nodes(input_number, output_number)
        for x in range(iterations):
            for row, target in zip(data, targets):
                self.update_weights(row, target)
        
        
    def activation_function(self, x):
        return 1 / (1 + exp(-x))
    
    def update_weights(self, row, target):
        values = [-1]       # Previous values plus bias node
        
        for value in row:
            values.append(value)
        
        for row_idx in range(len(self.hidden_nodes)):
            weights = []
            for node_idx in range(len(self.hidden_nodes[row_idx])):
                weights.append(self.hidden_nodes[row_idx][node_idx].weights)
            outputs = np.dot(weights, values)
            values = [-1]
            for idx in range(len(outputs)):
                self.hidden_nodes[row_idx][idx].value = \
                self.activation_function(outputs[idx])
                values.append(self.hidden_nodes[row_idx][idx].value)
                
        weights = []
        
        for node in self.output_nodes:
            weights.append(node.weights)
        outputs = np.dot(weights,values)
        values = []
        for idx in range(len(outputs)):
            self.output_nodes[idx].value = \
            self.activation_function(outputs[idx])
            values.append(self.output_nodes[idx].value)
        
        expected = np.zeros(len(self.targets))
        expected[self.targets.index(target)] = 1
        
        weights = []
        errors = []
        for idx in range(len(outputs)):
            self.output_nodes[idx].error = self.output_nodes[idx].value \
            * (1 - self.output_nodes[idx].value) * \
            (self.output_nodes[idx].value - expected[idx])
            #print("Error: ", self.output_nodes[idx].error)
            weights.append(self.output_nodes[idx].weights)
            errors.append(self.output_nodes[idx].error)
        
        error_sums = np.dot(errors, weights)
        #print (error_sums)
        
        for row_idx in range(len(self.hidden_nodes) - 1, -1, -1):
            errors = []
            weights = []
            for idx in range(len(error_sums) - 1):
                self.hidden_nodes[row_idx][idx].error = \
                self.hidden_nodes[row_idx][idx].value \
                * (1 - self.hidden_nodes[row_idx][idx].value) \
                * error_sums[idx + 1]
                errors.append(self.hidden_nodes[row_idx][idx].error)
            for node_idx in range(len(self.hidden_nodes[row_idx])):
                weights.append(self.hidden_nodes[row_idx][node_idx].weights)
            if (row_idx > 0):
                error_sums = np.dot(errors, weights)
                
        values = [-1]       # Previous values plus bias node
        
        for value in row:
            values.append(value)
        
        for row_idx in range(len(self.hidden_nodes)):
            for node_idx in range(len(self.hidden_nodes[row_idx])):
                for idx in range(len(self.hidden_nodes[row_idx][node_idx].weights)):
                    self.hidden_nodes[row_idx][node_idx].weights[idx] = \
                    self.hidden_nodes[row_idx][node_idx].weights[idx] \
                    - self.learning_rate * (self.hidden_nodes[row_idx][node_idx].error\
                                            * values[idx])
            values = [-1]
            for idx in range(len(self.hidden_nodes[row_idx])):
                values.append(self.hidden_nodes[row_idx][idx].value)
                
        for node_idx in range(len(self.output_nodes)):
            for idx in range(len(self.output_nodes[node_idx].weights)):
                self.output_nodes[node_idx].weights[idx] = \
                self.output_nodes[node_idx].weights[idx] - \
                self.learning_rate * (self.output_nodes[node_idx].error \
                                      * values[idx])
                
        
    
    def calculate_row(self, row):
        values = [-1]       # Previous values plus bias node
        
        for value in row:
            values.append(value)
        
        for row_idx in range(len(self.hidden_nodes)):
            weights = []
            for node_idx in range(len(self.hidden_nodes[row_idx])):
                weights.append(self.hidden_nodes[row_idx][node_idx].weights)
            outputs = np.dot(weights, values)
            values = [-1]
            for idx in range(len(outputs)):
                self.hidden_nodes[row_idx][idx].value = \
                self.activation_function(outputs[idx])
                values.append(self.hidden_nodes[row_idx][idx].value)
        
        weights = []
        for node in self.output_nodes:
            weights.append(node.weights)
        outputs = np.dot(weights,values)
        values = []
        for idx in range(len(outputs)):
            self.output_nodes[idx].value = \
            self.activation_function(outputs[idx])
            values.append(self.output_nodes[idx].value)
        
        biggest = values.index(max(values))
        #print ("Biggest: ", biggest)
        return self.targets[biggest]
        
    def predict(self, data):
        guesses = []
        for row in data:
            guesses.append(self.calculate_row(row))
        return guesses


# Read in the data from a file, convert the values into numbers, and normalize
# the data using Z-scores
def getCarEvaluationData():
    df = pd.read_csv('car.data', sep=",", header=None)
    df.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety",
                    "classification"]
    replace_values = {"buying"  : {"vhigh": 4, "high": 3, "med" :2, "low": 1},
                      "maint"   : {"vhigh": 4, "high": 3, "med" :2, "low": 1},
                      "doors"   : {"5more": 5,  "4": 4, "3": 3, "2": 2},
                      "persons" : {"more"  : 6, "4": 4, "2": 2},
                      "lug_boot" : {"small" : 1, "med" : 2, "big" :3},
                      "safety"  : {"low"   : 1, "med" : 2, "high":3}}
    df.replace(replace_values, inplace=True)
    df.head()
    cols = list(df.columns)
    cols.remove('classification')
    # Get the Z-scores for all columns except the classification column
    for col in cols:
        df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)
    data = df.values[:,0:6]
    target = df.values[:,6]

    return data, target;



# Get the data and targets
car_data, car_target = getCarEvaluationData()


# Divide up the training data and the testing data
train_data, test_data, train_target, test_target = train_test_split\
(car_data, car_target, test_size=0.3)

# These are some of the common best values from the code below
best_hidden_columns = 2
number_of_nodes = 4
best_learning_rate = .2
best_iterations = 38

# This part checks several different variables to see how they affect the
# accuracy, it is commented out because it takes a good amount of time to
# run it
'''
best_accuracy = 0
number_of_nodes = 0

print ("Changing the number of nodes in hidden columns")
for x in range(1,5):
    classifier = NeuralNetClassifier(hidden_numbers = [x, x], learning_rate=.2)
    classifier.fit(train_data,train_target, iterations=10)
    targets_predicted = classifier.predict(test_data)

    total = len(test_data)

    correct = 0

    for value1, value2 in zip(targets_predicted,test_target):
        if value1 == value2:
            correct += 1

    percent_correct = correct / total * 100

    print ("Accuracy ", x, ": ", percent_correct, " %")
    if (percent_correct > best_accuracy):
        best_accuracy = percent_correct
        number_of_nodes = x

best_accuracy = 0
best_iterations = 0
print ("Changing the number of iterations")
for x in range(1,20,2):
    classifier = NeuralNetClassifier(hidden_numbers = [number_of_nodes, \
                                                       number_of_nodes], \
                    learning_rate=.2)
    classifier.fit(train_data,train_target, iterations=2 * x)
    targets_predicted = classifier.predict(test_data)

    total = len(test_data)

    correct = 0

    for value1, value2 in zip(targets_predicted,test_target):
        if value1 == value2:
            correct += 1

    percent_correct = correct / total * 100

    print ("Accuracy ", 2 * x, ": ", percent_correct, " %")
    if (percent_correct > best_accuracy):
        best_accuracy = percent_correct
        best_iterations = 2 * x

best_accuracy = 0
best_learning_rate = 0
print ("Changing the learning rate")
for x in range(1,5):
    classifier = NeuralNetClassifier(hidden_numbers = [number_of_nodes, \
                                                       number_of_nodes], \
        learning_rate=.1 * x)
    classifier.fit(train_data,train_target, iterations=best_iterations)
    targets_predicted = classifier.predict(test_data)

    total = len(test_data)

    correct = 0

    for value1, value2 in zip(targets_predicted,test_target):
        if value1 == value2:
            correct += 1

    percent_correct = correct / total * 100

    print ("Accuracy ", .1 * x, ": ", percent_correct, " %")
    if (percent_correct > best_accuracy):
        best_accuracy = percent_correct
        best_learning_rate = .1 * x

best_accuracy = 0
best_hidden_columns = []
print ("Changing the number of hidden columns")
for x in range(1,5):
    hidden_numbers = np.full(x,number_of_nodes)
    classifier = NeuralNetClassifier(hidden_numbers, learning_rate=best_learning_rate)
    classifier.fit(train_data,train_target, iterations=best_iterations)
    targets_predicted = classifier.predict(test_data)

    total = len(test_data)

    correct = 0

    for value1, value2 in zip(targets_predicted,test_target):
        if value1 == value2:
            correct += 1

    percent_correct = correct / total * 100

    print ("Accuracy ", x, ": ", percent_correct, " %")
    if (percent_correct > best_accuracy):
        best_accuracy = percent_correct
        best_hidden_columns = x
'''
hidden_numbers = np.full(best_hidden_columns,number_of_nodes)
classifier = NeuralNetClassifier(hidden_numbers, learning_rate=best_learning_rate)
classifier.fit(train_data,train_target, iterations=best_iterations)
targets_predicted = classifier.predict(test_data)

total = len(test_data)

correct = 0

for value1, value2 in zip(targets_predicted,test_target):
    if value1 == value2:
        correct += 1

percent_correct = correct / total * 100

print ("Accuracy: ", percent_correct, " %")

        