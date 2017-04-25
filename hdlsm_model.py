import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


# d denotes document index
# i denotes an observation in a document
# m denotes a motif
# o denotes a motif occurrence in a document

class HierarchicalDirichletLatentSemanticMotifs:
    def __init__(self, motif_length, n_words, alpha, eta, gamma, niter=100, initial_motifs=None):
        self.wo = []  # d, i
        self.om = []  # d, o
        self.ost = []  # d, o

        self.motif_length = motif_length
        self.n_words = n_words

        self.alpha = alpha
        self.eta = eta
        self.gamma = gamma

        self.niter = niter

        if initial_motifs is None:
            self.motifs = []
        else:
            self.motifs = [m.copy() for m in initial_motifs]

        self.docs_ = None

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
            n_obs_d = self.wo[d].shape[0]
            n_ts_d = numpy.max(docs[d][:, 1]) + 1
            for i in range(n_obs_d):
                w_di = docs[d][i, 0]
                t_di = docs[d][i, 1]
                old_wo_di = self.wo[d][i]
                n_occ_d = len(self.om[d])  # number of occurrences in the current doc

                self._cancel_obs(d, i)

                probas = numpy.zeros((n_occ_d + 1, ))
                # Existing occurrence case
                for o in range(n_occ_d):
                    rt_di = t_di - self.ost[d][o]
                    if 0 <= rt_di < self.motif_length:
                        m_di = self.om[d][o]
                        # Using Eq. 10 from supp material (cf pdf)
                        p_wt = (self._n_obs_wtm(w_di, rt_di, m_di) + self.eta) / (self._sum_wt_n_obs_wtm(m_di) + self.n_words * self.motif_length * self.eta)  # Assumes uniform prior
                        # Using Eq. 11 from supp material (cf pdf)
                        p_o = self._n_obs_do(d, o) / (n_obs_d - 1 + self.alpha)
                        probas[o] = p_wt * p_o
                # New occurrence case
                probas_motif = numpy.zeros((self.n_motifs + 1, ))
                denom_gamma = self._n_occ() + self.gamma  # Denom in Eq 15
                p_o = self.alpha / (n_obs_d - 1 + self.alpha)
                for m in range(self.n_motifs):
                    p_wt = (self._sum_t_n_obs_wtm(w_di, m) + self.eta) / (self._sum_wt_n_obs_wtm(m) + self.eta * self.n_words * self.motif_length)  # Eq 16
                    p_k = self._n_occ_m(m) / denom_gamma  # Eq 15
                    probas_motif[m] = p_k * p_wt / n_ts_d # Eq 14
                # New occurrence, new motif case
                probas_motif[-1] = self.gamma / (denom_gamma * self.n_words * n_ts_d)  # p_o factor is given afterwards
                probas[-1] = p_o * numpy.sum(probas_motif)

                new_wo_di = numpy.random.multinomial(1, pvals=probas / numpy.sum(probas))
                if new_wo_di == n_occ_d:
                    self.ost[d].append(t_di - numpy.random.randint(self.motif_length))
                    self.om[d].append(numpy.random.multinomial(1, pvals=probas_motif / numpy.sum(probas_motif)))
                self._change_occurrence(d, i, old_wo_di, new_wo_di)

    def _n_obs_wtm(self, w, rt, m):
        n_obs = 0
        for d in range(self.n_docs):
            indices = self.docs_[d][:, 0] == w
            absolute_times = self.docs_[d][indices, 1]
            occurrences = self.wo[d][indices]
            for o, at in zip(occurrences, absolute_times):
                if numpy.isfinite(o) and self.om[d][o] == m and (at - self.ost[d][o]) == rt:
                    n_obs += 1
        return n_obs

    def _sum_t_n_obs_wtm(self, w, m):
        n_obs = 0
        for d in range(self.n_docs):
            indices = self.docs_[d][:, 0] == w
            occurrences = self.wo[d][indices]
            for o in occurrences:
                if numpy.isfinite(o) and self.om[d][o] == m:
                    n_obs += 1
        return n_obs

    def _sum_wt_n_obs_wtm(self, m):
        n_obs = 0
        for d in range(self.n_docs):
            for o in self.wo[d]:
                if numpy.isfinite(o) and self.om[d][o] == m:
                    n_obs += 1
        return n_obs

    def _n_occ(self):
        n_occ = 0
        for d in range(self.n_docs):
            n_occ += len(self.om[d])
        return n_occ

    def _n_occ_m(self, m):
        n_occ_m = 0
        for d in range(self.n_docs):
            for _m in self.om[d]:
                if _m == m:
                    n_occ_m += 1
        return n_occ_m

    def _n_obs_do(self, d, o):
        return numpy.sum(self.wo[d] == o)

    def _cancel_obs(self, d, i):
        self.wo[d][i] = numpy.nan

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
        self.wo = [numpy.zeros((doc_d.shape[0], ), dtype=numpy.int) * numpy.nan for doc_d in docs]
        self.om = [[] for doc_d in docs]
        self.ost = [[] for doc_d in docs]
        self.docs_ = docs