import numpy as np
import pandas as pd


# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


def find_best_attribute(X, y, attributes):
    """
    Finds the best attribute to split on

    Args:
        X (pd.DataFrame): a matrix with discrete value where
            each row is a sample and the columns correspond
            to the features.
        y (pd.Series): a vector of discrete ground-truth labels
        attributes (list<str>): a list of all attributes

    Returns:
        The name of the attribute to split on
    """
    # Compute entropy for each attribute
    entropies = []
    for attr in attributes:
        # Compute entropy for each value of attribute
        entropy_attr = 0
        for value in X[attr].unique():
            # Get subset of data
            X_subset = X[X[attr] == value]
            y_subset = y[X_subset.index]

            # Compute entropy
            entropy_attr += len(X_subset) / len(X) * entropy(y_subset.value_counts())

        # Store entropy for attribute
        entropies.append(entropy_attr)

    # Return attribute with smallest entropy
    return attributes[np.argmin(entropies)]


class Node:
    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value = value
        self.children = []

    def add_child(self, node):
        self.children.append(node)

    def __repr__(self):
        return f"Node({self.attribute}, {self.value}, {self.children})"


class DecisionTree:

    def __init__(self, default=None):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.tree: Node | None = None
        self.default = default

    def id3(self, X, y, attributes):
        """
        ID3 algorithm

        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
            attributes (list<str>): a list of all attributes

        Returns:
            A decision tree
        """
        # Initialize root node
        tree = Node(None, None)

        # If all samples are of the same class, return the class
        if len(y.unique()) == 1:
            tree.value = y.iloc[0]
            return tree

        # If attributes is empty, return the root node with majority class
        if len(attributes) == 0:
            tree.value = y.value_counts().idxmax()
            return tree

        # Find best attribute to split on
        best_attr = find_best_attribute(X, y, attributes)

        # Split on best attribute
        for value in X[best_attr].unique():
            # Split data
            X_subset = X[X[best_attr] == value]  # new branch
            tree.add_child(Node(best_attr, value))
            y_subset = y[X_subset.index]  # subset of examples
            if len(X_subset) == 0:  # Examples empty, add leaf node with majority class
                tree.children[-1].add_child(Node(None, y.value_counts().idxmax()))
            else:
                # Remove attribute from list of attributes
                attributes_subset = attributes.copy()
                attributes_subset.remove(best_attr)

                # Recursively build tree
                tmp = self.id3(X_subset, y_subset, attributes_subset)
                if tmp.value is not None:
                    tree.children[-1].add_child(tmp)
                else:
                    for child in tmp.children:
                        tree.children[-1].add_child(child)
        return tree

    def fit(self, X, y):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        # ID3 algorithm
        self.tree = self.id3(X, y, X.columns.tolist())

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """

        res = np.array([])
        for row in X.itertuples():
            # Copy tree
            tree: Node = self.tree

            # Traverse tree
            while tree.children:
                for child in tree.children:
                    # Get index of attribute
                    if not child.attribute:
                        tree = child
                        break
                    if row[X.columns.get_loc(child.attribute) + 1] == child.value:
                        tree = child
                        break
                else:  # No matching value found, use default
                    tree = Node(None, self.default)

            # Add prediction
            res = np.append(res, [tree.value])

        return res

    def _get_rules(self, tree, rules, cur_rule):
        """
        Recursively traverses the decision tree and returns
        a list of rules

        Returns:
            A list of rules
        """
        if not tree.children:  # leaf node
            rules[-1] = (rules[-1], tree.value)
            return rules

        # Add rule
        cr = None
        if tree.attribute:
            cr = cur_rule.copy() + [(tree.attribute, tree.value)]
            if not rules or type(rules[-1]) is tuple:
                rules.append(cr)
            else:
                rules[-1] = cr
        for child in tree.children:  # recursively traverse tree pre-order
            self._get_rules(child, rules, cr if cr else cur_rule)
        return rules

    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        return self._get_rules(self.tree, [], [])  # remove last empty rule


# --- Some utility functions 

def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a length k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))
