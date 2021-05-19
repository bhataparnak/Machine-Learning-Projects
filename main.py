from decision_tree_learner import *
from ID3Algo_entropy import *
import sys
import os.path
import csv


def get_data(filename, attributes):
    with open(filename, 'r') as f:
        lines = [row for row in csv.reader(f.read().splitlines())]
    data = []
    for line in lines:
        data.append(dict(zip(attributes, [datum.strip() for datum in line])))

    return data


def print_tree(tree, str):
    if type(tree) == dict:
        trial = list(tree.keys())
        print("%s%s" % (str, trial[0]))
        trial1 = list(tree.values())
        for item in trial1[0].keys():
            print("%s\t%s" % (str, item))
            print_tree(trial1[0][item], str + "\t")
    else:
        print("%s\t->\t%s" % (str, tree))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        training_filename = input("Training Filename: ")
        test_filename = input("Test Filename: ")
    else:
        training_filename = sys.argv[1]
        test_filename = sys.argv[2]

    def file_exists(filename):
        if os.path.isfile(filename):
            return True
        else:
            print("Error: The file '%s' does not exist." % filename)
            return False

    if ((not file_exists(training_filename)) or
            (not file_exists(test_filename))):
        sys.exit(0)

    attributes = ["A", "B", "C", "D", "E",
                  "F", "G", "H", "I", "Decision"]
    target_attr = attributes[9]

    training_data = get_data(training_filename, attributes)
    test_data = get_data(test_filename, attributes)

    dtree = build_decision_tree(training_data, attributes, target_attr, info_gain)
    print(dtree)

    print("\nDecision Tree\n")
    print_tree(dtree, "")

    classification = classify(dtree, test_data)

    print("\nClassification\n")
    for item in classification:
        print(item)
