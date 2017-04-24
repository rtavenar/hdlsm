import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


# d denotes document index
# i denotes an observation in a document
# o denotes a motif occurrence in a document

class HierarchicalDirichletLatentSemanticMotifs:
    def __init__(self, topic_length, n_words, initial_motifs=None):
        self.wo = []  # d, i
        self.oz = []  # d, o
        self.ost = []  # d, o

        self.topic_length = topic_length
        self.n_words = n_words

        if initial_motifs is None:
            self.motifs = []
        else:
            self.motifs = [m.copy() for m in initial_motifs]

    def fit(self, docs):
        self.__init_from_data(docs)

    def __init_from_data(self, docs):
        self.wo = [numpy.zeros((doc_d.shape[0], ), dtype=numpy.int) - 1 for doc_d in docs]
        self.oz = [[] for doc_d in docs]
        self.ost = [[] for doc_d in docs]