# Reference Credits
# 1) https://www.python-course.eu/Decision_Trees.php
# 2) https://dhirajkumarblog.medium.com/decision-tree-from-scratch-in-python-629631ec3e3a


import collections

# Creates a list of all values in the target attribute for each record
# in the data list object, and returns the value that appears in this list
# the most frequently.


def major_val(data, target_attr):

    data = data[:]
    return most_freq([record[target_attr] for record in data])


def most_freq(lst):
    # Returns the item that appears most frequently in the given list.

    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in distinct(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)

    return most_freq

# Returns a list made up of the unique values found in lst.  i.e., it
    # removes the redundant values in lst.


def distinct(lst):
    lst = lst[:]
    distinct_lst = []
    # Cycle through the list and add each value to the unique list only once.
    for item in lst:
        if distinct_lst.count(item) <= 0:
            distinct_lst.append(item)
            # Return the list with all redundant values removed.
    return distinct_lst

# Creates a list of values in the chosen attribut for each record in data,
    # prunes out all of the redundant values, and return the list.


def get_values(data, attr):
    data = data[:]
    return distinct([record[attr] for record in data])

# Cycles through all the attributes and returns the attribute with the
    # highest information gain (or lowest entropy).


def select_attr(data, attributes, target_attr, fitness):
    data = data[:]
    best_gain = 0.0
    best_attr = None

    for attr in attributes:
        info_gain = fitness(data, attr, target_attr)
        if (info_gain >= best_gain and attr != target_attr):
            best_gain = info_gain
            best_attr = attr

    return best_attr

# Returns a list of all the records in <data> with the value of <attr>
    # matching the given value.


def get_samples(data, attr, value):
    data = data[:]
    rtn_lst = []

    if not data:
        return rtn_lst
    else:
        record = data.pop()
        if record[attr] == value:
            rtn_lst.append(record)
            rtn_lst.extend(get_samples(data, attr, value))
            return rtn_lst
        else:
            rtn_lst.extend(get_samples(data, attr, value))
            return rtn_lst

# This function recursively traverses the decision tree and returns a
    # classification for the given record.


def get_classification(record, tree):
    if type(tree) == type("string"):
        return tree
    else:
        attr = list(tree.keys())
        attr1 = attr[0]
        t = tree[attr1][record[attr1]]
        return get_classification(record, t)

# Returns a list of classifications for each of the records in the data
    # list as determined by the given decision tree.


def classify(tree, data):
    data = data[:]
    classification = []

    for record in data:
        classification.append(get_classification(record, tree))

    return classification

# Returns a new decision tree based on the examples given.


def build_decision_tree(data, attributes, target_attr, fitness_func):
    data = data[:]
    vals = [record[target_attr] for record in data]
    default = major_val(data, target_attr)
# If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.
    if not data or (len(attributes) - 1) <= 0:
        return default
        # If all the records in the dataset have the same classification,
    # return that classification.
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        # Choose the next best attribute to best classify our data
        best = select_attr(data, attributes, target_attr,
                                fitness_func)

        tree = {best: collections.defaultdict(lambda: default)}
 # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for val in get_values(data, best):
            # Create a subtree for the current value under the "best" field
            subtree = build_decision_tree(
                get_samples(data, best, val),
                [attr for attr in attributes if attr != best],
                target_attr,
                fitness_func)
            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            tree[best][val] = subtree

    return tree
