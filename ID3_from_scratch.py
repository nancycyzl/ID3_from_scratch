import pandas as pd
import math


def calculate_entropy(y):  # y is a list
    total_number = len(y)
    unique_values = set(y)
    count = [y.count(one_level) for one_level in unique_values]
    prob = [one_count / total_number for one_count in count]
    entropy = sum([-one_prob * math.log(one_prob, 2) for one_prob in prob])

    return entropy


def calculate_IG(dataset, feature):
    orig_entropy = calculate_entropy(list(dataset[dataset.columns[-1]].values))
    feature_values = dataset[feature].unique()
    sum_entropy = 0
    for one_value in feature_values:
        subset = dataset[dataset[feature] == one_value]
        subset_entropy = len(subset) / len(dataset) * calculate_entropy(list(subset[subset.columns[-1]].values))
        sum_entropy += subset_entropy

    return orig_entropy - sum_entropy


def get_best_feature(dataset, features, verbose=True):
    IG_list = [calculate_IG(dataset, feature) for feature in features]
    largest_IG_index = IG_list.index(max(IG_list))
    if verbose:
        print("Calculate IG for each feature")
        print(features)
        print(IG_list)

    return features[largest_IG_index]


class Node:
    def __init__(self):
        self.value = None

class DecisionTreeID3:
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label
        self.root = None

    def build_tree(self, verbose=True):
        features = list(self.dataset.columns.values)
        features.remove(self.label)
        if verbose:
            print("All features", end=" ")
            print(features)
        self.root = self.build_recv(self.root, self.dataset, features, self.label, verbose)

    def build_recv(self, root, dataset, features, label, verbose):
        if verbose:
            print("labels for each instance is:")
            print(dataset[label].value_counts().to_string())
        if not root:
            root = Node()
            # print("A new node is made")

        if dataset[label].nunique() == 1:
            root.value = list(dataset[label].values)[0]
            print("A new node is made, value is", root.value)
            return root
        elif len(features) == 0:
            target_values = dataset[label]
            root.value = max(target_values, key=target_values.count)
            print("A new node is made, value is", root.value)
            return root
        else:
            best_feature = get_best_feature(dataset, features,
                                            verbose)  # will print IG for each feature is verbose=True
            print("A new node is made, value is", best_feature)
            root.value = best_feature
            feature_values = dataset[best_feature].unique()
            if verbose:
                print("best_feature values", end=" ")
                print(feature_values)
            for value in feature_values:
                child = Node()
                print(best_feature + ": " + str(value))
                subset = dataset[dataset[best_feature] == value]
                new_features = [feature for feature in features if feature != best_feature]
                if verbose:
                    print(subset)
                    print("after remove best feature, the feature list is")
                    print(new_features)
                child = self.build_recv(child, subset, new_features, label, verbose)
            return root


table = pd.read_excel("example_data.xlsx")
# print(table, end='\n')

clf = DecisionTreeID3(table, label="play")
clf.build_tree(verbose=False)
