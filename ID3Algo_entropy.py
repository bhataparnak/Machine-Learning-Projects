# Reference Credits
# 1) https://www.python-course.eu/Decision_Trees.php
# 2) https://dhirajkumarblog.medium.com/decision-tree-from-scratch-in-python-629631ec3e3a

import math

# Calculates the entropy and frequency of the given data set for the target attribute.
def cal_entropy(data, target_attr):
    value_freq = {}
    data_entropy = 0.0
    for record in data:
        # Calculate the frequency of each of the values in the target attr
        if record[target_attr] in value_freq:
            value_freq[record[target_attr]] += 1.0
        else:
            value_freq[record[target_attr]] = 1.0

# Calculate the entropy of the data for the target attribute
    for freq in value_freq.values():
        data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2)
    return data_entropy


# Calculates the information gain (reduction in entropy) that would
# result by splitting the data on the chosen attribute (attr).

def info_gain(data, attr, target_attr):
    value_freq = {}
    sub_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for record in data:
        if record[attr] in value_freq:
            value_freq[record[attr]] += 1.0
        else:
            value_freq[record[attr]] = 1.0
    
    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    print("For attribute '%s' \nEach value frequency is :%s " % (attr, value_freq))
    for value in value_freq.keys():
        value_prob = value_freq[value] / sum(value_freq.values())
        data_subset = [record for record in data if record[attr] == value]
        sub_entropy += value_prob * cal_entropy(data_subset, target_attr)
 # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
        entropy_gain = (cal_entropy(data, target_attr) - sub_entropy)
        print("For Attribute : %s" % attr)
        print("Entropy Gain: %f" % entropy_gain)
    return entropy_gain
