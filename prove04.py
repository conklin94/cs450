#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 19:29:35 2019

@author: sethconklin
"""

from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from graphviz import Digraph
import pandas as pd
import numpy as np

def calc_entropy(p):
    if p != 0:
        return -p * np.log2(p)
    else:
        return 0

class Node:
    def __init__(self, value=None):
        self.value = value
        self.children = []
        self.branches = []
        
    def set_child(self, node, branch):
        self.children.append(node)
        self.branches.append(branch)


class ID3Classifier:
    def __init__(self):
        self.root = None
        self.tree = None
        self.index = 0
                
    def most_common(self, target):
        counter = Counter(target)
        return counter.most_common(1)[0][0]
                
    
    def make_tree(self, data, target, columns):
        if (len(set(target)) == 1):
            return Node(value=target[0])
        # This checks if the number of unused columns is 1 or less instead
        # of zero because it helps diminish overfitting
        elif (len(columns) < 2):
            return Node(value=self.most_common(target))
        else:
            size = len(data)
            best_values = dict()
            best_column = None
            best_entropy = None
            for column in columns:
                column_entropy = 0
                current_column = data[:,column]
                values = dict()
                for x in range(len(current_column)):
                    if current_column[x] in values:
                        if target[x] in values[current_column[x]]:
                            values[current_column[x]][target[x]] += 1
                            values[current_column[x]]['total'] += 1
                        else:
                            values[current_column[x]][target[x]] = 1
                            values[current_column[x]]['total'] += 1
                    else:
                        values[current_column[x]] = {target[x]: 1, 'total': 1}
                
                for key, value in values.items():
                    entropy = 0
                    current_total = values[key]['total']
                    for key2, value2 in value.items():
                        if (key2 != 'total'):    
                            entropy += calc_entropy(value2 / current_total)
                    column_entropy += entropy * current_total / size
                if (best_entropy == None):
                    best_entropy = column_entropy
                    best_column = column
                    best_values = values
                elif (column_entropy < best_entropy):
                    best_entropy = column_entropy
                    best_column = column
                    best_values = values
                    
            node = Node(best_column)
            new_columns = columns[:]
            new_columns.remove(best_column)
            for key, value in best_values.items():
                new_data = []
                new_target = []
                for x in range(size):
                    if (data[x][best_column] == key):
                        new_data.append(data[x])
                        new_target.append(target[x])
                node.set_child(self.make_tree(np.array(new_data),
                                              np.array(new_target),
                                              new_columns),key)
            return node
        
    def visualizeTree(self, node, parent=None):
        if parent == None:
            parent = 'N' + str(self.index) + '=' + str(node.value)
            self.tree.node(parent)
        self.index += 1;
        for branch, child in zip(node.branches, node.children):
            currentBranch = str(branch)
            currentNode = 'N' + str(self.index) + '=' + str(child.value)
            self.tree.node(currentNode)
            self.tree.edge(parent, currentNode, label=currentBranch)
            self.visualizeTree(child,currentNode)
            self.index += 1
        


    def displayTree(self):
        self.tree = Digraph('ID3_tree_visualization', filename='id3.gv')
        self.tree.attr('node', shape='circle')
        self.visualizeTree(self.root)
        self.tree.view()
            
    def findValue(self, node, row):
        if (not node.children):
            return node.value
        else:
            current_value = row[node.value]
            for x in range(len(node.branches)):
                if (node.branches[x] == current_value):
                    return self.findValue(node.children[x], row)
                
        
    def fit(self, data, target):
        data_copy = data
        target_copy = target
        columns = list(range(0,len(data[0])))
        self.root = self.make_tree(data_copy, target_copy, columns)
        
    def predict(self, data):
        guesses = []
        for row in data:
            guesses.append(self.findValue(self.root,row))
        return guesses


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

    data = df.values[:,0:6]
    target = df.values[:,6]
 
    return data, target;

data, target = getCarEvaluationData()


train_data, test_data, train_target, test_target = train_test_split\
(data, target, test_size=0.3)

total = len(test_target)


classifier = ID3Classifier()
classifier.fit(train_data,train_target)

classifier.displayTree()

classifier2 = DecisionTreeClassifier()
classifier2.fit(train_data,train_target)

targets_predicted = classifier.predict(test_data)
targets_predicted2 = classifier2.predict(test_data)

correct = 0
correct2 = 0


for test1, test2, target in zip(targets_predicted, targets_predicted2,
                                test_target):
    if test1 == target:
        correct += 1
    if test2 == target:
        correct2 += 1

        
percent_correct = correct / total * 100
percent_correct2 = correct2 / total * 100

print ("My ID3 Accuracy: ", percent_correct, "%")
print ("DecisionTreeClassifier Accuracy: ", percent_correct2, "%")