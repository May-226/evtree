import numpy as np

class evtree(object):


    def __init__(self, p_crossover=0.6, p_mutation=0.4, p_split=0.1, p_prune=0.1, population_size = 400, max_iter = 500):
        self.population = [[] for i in range(population_size)]
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

    def initialization(self, X, y):
        for _ in xrange(self.population_size):

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
