import random
from main import *
import matplotlib.pyplot as plt


def generate_dataSet(lines, attributes):
    data = []
    for line in lines:
        data.append(dict(zip(attributes, [datum.strip() for datum in line])))
    return data


def majority_Voting(List):
    counter = {}
    for Class in List:
        targetClass = Class[0]
        if targetClass in counter:
            counter[targetClass] += 1
        else:
            counter[targetClass] = 1
    popularClass = sorted(counter, key=counter.get, reverse=True)
    top_1 = popularClass[:1]
    return top_1


with open("tic-tac-toe_train.csv", 'r') as f:
    lines = [row for row in csv.reader(f.read().splitlines())]

trainSize = len(lines)

ListOfClassifiers = []

attributes = ["A", "B", "C", "D", "E",
              "F", "G", "H", "I", "Decision"]
target_attr = attributes[9]


for i in range(10):
    baggingDataSet = []
    dataSet = []
    for x in range(trainSize):
        index = random.randrange(0, trainSize-1, 1)
        dataSet.append(lines[index])
    baggingDataSet = generate_dataSet(dataSet, attributes)
    tree = build_decision_tree(baggingDataSet, attributes, target_attr, info_gain)
    ListOfClassifiers.append(tree)

# read test data
with open("tic-tac-toe_test.csv", 'r') as f:
    lines = [row for row in csv.reader(f.read().splitlines())]
test_data = generate_dataSet(lines, attributes)

# Predict Class value for each Test data with N Bagging Data sets
MostFreqClass = []
for data in test_data:
    Result = []
    data_list = [data]
    for classifier in ListOfClassifiers:
        Result.append(classify(classifier, data_list))
    MostFreqClass.append(majority_Voting(Result))
for i in MostFreqClass:
    print(i[0])



