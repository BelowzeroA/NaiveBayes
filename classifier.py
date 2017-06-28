from __future__ import division
from collections import defaultdict
from math import log


class NBClassifier:

    def __init__(self):
        self.classes = defaultdict(lambda: 0)
        self.freq = defaultdict(lambda: 0)

    def train(self, samples):

        for feats, label in samples:
            self.classes[label] += 1  # count classes frequencies
            for feat in feats:
                self.freq[label, feat] += 1  # count features frequencies

        for label, feat in self.freq:  # normalize features frequencies
            self.freq[label, feat] /= self.classes[label]
        for c in self.classes:  # normalize classes frequencies
            self.classes[c] /= len(samples)

        #return classes, freq  # return P(C) and P(O|C)


    def classify(self, feats):
        #classes, prob = classifier
        # calculate argmin(-log(C|O))
        return min(self.classes.keys(),
                   key = lambda cl: -log(self.classes[cl]) + \
                                  sum(-log(self.freq.get((cl, feat), 10 ** (-7))) for feat in feats))
