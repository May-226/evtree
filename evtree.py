import numpy as np
import csv
import math

class Node(object):
    def __init__(self, attribute=None, value=None, terminal_node=False, intermediate=False):
        self.attribute = attribute
        self.value = value
        self.left = None
        self.right = None
        self.terminal_node = terminal_node
        self.intermediate = intermediate


class DecisionTree(object):
    def __init__(self):
        self.tree = None

    def createTree(self, X, y):
        pass

class evtree(object):


    def __init__(self, p_crossover=0.6, p_mutation=0.4, p_split=0.1, p_prune=0.1, population_size = 400, max_iter = 500):
        self.population = [[None]]*population_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.p_split = p_split
        self.p_prune = p_prune
        self.population_size = population_size
        self.max_iter = max_iter

        self.num_attributes = None

        self.best_candidates = []

    def fit(self, X, y):
        self.num_attributes = X.shape[1]

        # initialization

        for tree_idx in xrange(self.population_size):
            self.split(tree_idx, X, y)


    def test(self, X):
        pass

    def treeDepth(self, tree):
        # given a tree, calculate its depth
        pass

    def split_points(self, values):
        # given a numpy array of values, calculate the split points.
        return (values[1:] + values[:-1])/2.0

    def get_threshold(self, attr_idx, X, y):
        # Given data and attributes, suggests a split point radomly
        vals = np.sort(X[:, attr_idx])
        thresholds = self.split_points(vals)

        threshold = np.random.choice(thresholds, 1)[0]
        return threshold


    def create_inital_tree(self, X, y, attributes):
        split_attribute = attributes.pop(np.random.randint(0, len(attributes)))
        split_value = self.get_threshold(split_attribute.keys()[0], X, y)



    def initialization(self, X, y):
        m, n = X.shape
        for idx in xrange(self.population_size):
            initial_tree = 0


        pass

    def evaluate(self, X, y):
        # given a decision tree, return the classification accuracy?
        pass

    def split(self, tree_idx, X, y):
        tree = self.population[tree_idx]
        if not tree:
            # choose attribute to split on
            attr_idx = np.random.randint(0, self.num_attributes)
            # choose the threshold
            attr_vals = X[:, attr_idx]
            min_val = np.min(attr_vals)
            threshold = min_val
            while threshold == min_val:
                threshold = np.random.choice(attr_vals, 1)[0]
            tree.append((attr_idx, threshold))

    def prune(self, tree_idx):
        pass

    def minor_mutate(self, tree_idx):
        pass

    def major_mutate(self, tree_idx):
        pass

    def crossover(self, tree_idx1, tree_idx2):
        pass

    def survivor_selction(self):
        pass

def quartile_bins(data):
    '''Converts a continuous variable into categorical variable with 4 outcomes
    Example Input: [3,1,4,5,6,8,2,10,9,7]
    Example Output: ['2.5-5','1-2.5','2.5-5',...,'5-7.5']'''
    a = [float(x) for x in data]
    b = a
    a = sorted(a)
    a = np.array(a)
    p25, p50, p75 = np.percentile(a, [25, 50, 75])
    acat = []
    for point in b:
        if (point <= p25):
            acat.append(str(a[0])+' - '+str(p25))
        elif (point <= p50):
            acat.append(str(p25)+' - '+str(p50))
        elif (point <= p75):
            acat.append(str(p50)+' - '+str(p75))
        else:
            acat.append(str(p75)+' - '+str(a[-1]))
    return acat

def data_preprocess(data):
    '''Preprocess features. Convert continuous variables to categorical variables'''
    features = len(data[0]) - 1
    data_points = len(data)
    dataX = []
    for i in range(features):
        vals = [rowset[i] for rowset in data]
        valcat = quartile_bins(vals)
        dataX.append(valcat)
    Ys = [rowset[features] for rowset in data]
    dataX.append(Ys)
    dataX = zip(*dataX)
    return dataX

if __name__ == '__main__':
    with open('hw4-data.csv') as f:
        header = next(f, None)

    data = np.genfromtxt('hw4-data.csv', delimiter=',')
    X = data[1:, :-1]
    y = data[1:, 0:1]

    attributes = [{idx: attr.strip()} for idx, attr in enumerate(header.split(","))]
    #print attributes
    outcome = attributes[-1].values()[0]
    attributes = attributes[:-1]

    et = evtree()
    #print et.get_threshold(2, X, y)