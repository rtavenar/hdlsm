import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


# d denotes document index
# i denotes an observation in a document
# m denotes a motif
# o denotes a motif occurrence in a document

class HierarchicalDirichletLatentSemanticMotifs:
    def __init__(self, motif_length, n_words, alpha, eta, mu, niter=100, initial_motifs=None):
        self.wo = []  # d, i
        self.om = []  # d, o
        self.ost = []  # d, o

        self.motif_length = motif_length
        self.n_words = n_words

        self.alpha = alpha
        self.eta = eta
        self.mu = mu

        self.niter = niter

        if initial_motifs is None:
            self.motifs = []
        else:
            self.motifs = [m.copy() for m in initial_motifs]

    @property
    def n_docs(self):
        return len(self.wo)

    @property
    def n_motifs(self):
        return len(self.motifs)

    def fit(self, docs):
        """Fitting the model to observations found in docs.

        docs is a list of numpy arrays. Each array has shape (n_obs[d], 2).
        docs[d][:, 0] contains words.
        docs[d][:, 1] contains timestamps.
        """
        self.__init_from_data(docs)
        for _ in range(self.niter):
            self._fit_one_iter(docs)
        return self

    def _fit_one_iter(self, docs):
        for d in range(self.n_docs):
            n_obs = self.wo[d].shape[0]
            n_ts = numpy.max(docs[d][:, 1]) + 1
            for i in range(n_obs):
                w_di = docs[d][i, 0]
                t_di = docs[d][i, 1]
                old_wo_di = self.wo[d][i]
                n_occ = len(self.om[d])  # number of occurrences in the current doc

                probas = numpy.zeros((n_occ + 1, ))
                # Existing occurrence case
                for o in range(n_occ):
                    rt_di = t_di - self.ost[d][o]
                    if 0 <= rt_di < self.motif_length:
                        m_di = self.om[d][o]
                        # Using Eq. 10 from supp material (cf pdf)
                        p_wt = (self._n_obs_wtm(w_di, rt_di, m_di) + self.eta) / (self._sum_n_obs_wtm(m_di) + self.n_words * self.motif_length * self.eta)  # Assumes uniform prior
                        # Using Eq. 11 from supp material (cf pdf)
                        p_o = self._n_obs_do(d, o) / (n_obs - 1 + self.alpha)
                        probas[o] = p_wt * p_o
                # New occurrence case
                probas_motif = numpy.zeros((self.n_motifs, ))
                denom_mu = self.occ__() + self.mu
                for m in range(self.n_motifs):
                    #p_wt =
                    #p_o = self.alpha / (n_obs - 1 + self.alpha)
                    probas_motif[m] = (self.occ_X(m) / denom_mu) * ((self.obsX_X(w_di, m) + self.motif_length * self.eta) / (self.obs__X(m) + self.motif_length * self.n_words * self.eta)) / n_ts
                # New occurrence, new motif case
                probas_motif[-1] = self.mu / (denom_mu * self.n_words * n_ts)
                probas[-1] = self.alpha * numpy.sum(probas_motif)

                new_wo_di = numpy.random.multinomial(1, pvals=probas / numpy.sum(probas))
                if new_wo_di == n_occ:
                    self.ost[d].append(t_di - numpy.random.randint(self.motif_length))
                    self.om[d].append(numpy.random.multinomial(1, pvals=probas_motif / numpy.sum(probas_motif)))
                self._change_occurrence(d, i, old_wo_di, new_wo_di)

    def _change_occurrence(self, d, i, old_wo_di, new_wo_di):
        self.wo[d][i] = new_wo_di
        if not self._exists_occurrence(d, old_wo_di):
            pass  # TODO: should include index remapping if the occurrence disappears

    def _exists_occurrence(self, d, o):
        for _o in self.wo[d]:
            if _o == o:
                return True
        return False

    def __init_from_data(self, docs):
        self.wo = [numpy.zeros((doc_d.shape[0], ), dtype=numpy.int) - 1 for doc_d in docs]
        self.om = [[] for doc_d in docs]
        self.ost = [[] for doc_d in docs]