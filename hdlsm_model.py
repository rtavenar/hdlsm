import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


# d denotes document index
# i denotes an observation in a document
# m denotes a motif
# o denotes a motif occurrence in a document

class HierarchicalDirichletLatentSemanticMotifs:
    def __init__(self, motif_length, n_words, alpha, eta, gamma, n_iter=100):
        self.wo = []  # d, i -> o
        self.om = []  # d, o -> m
        self.ost = []  # d, o -> st

        self.motif_length = motif_length
        self.n_words = n_words

        self.alpha = alpha
        self.eta = eta
        self.gamma = gamma

        self.n_iter = n_iter

        self.docs_ = None

        self.n_occ_ = 0
        self.n_occ_m_ = []

        self.n_obs_m_ = []
        self.n_obs_wtm_ = []

    @property
    def n_docs(self):
        return len(self.wo)

    @property
    def n_motifs(self):
        return len(self.n_occ_m_)

    # Short names for n_words and motif_length
    @property
    def n_t(self):
        return self.motif_length

    @property
    def n_w(self):
        return self.n_words

    def p_wt_m(self, m):
        return (self.n_obs_wtm_[m] + self.eta) / (self.n_obs_m_[m] + self.n_w * self.n_t * self.eta)

    def fit(self, docs):
        """Fitting the model to observations found in docs.

        docs is a list of numpy arrays. Each array has shape (n_obs[d], 2).
        docs[d][:, 0] contains words.
        docs[d][:, 1] contains timestamps.
        """
        self.__init_from_data(docs)
        for _ in range(self.n_iter):
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
                    if 0 <= rt_di < self.n_t:
                        m_di = self.om[d][o]
                        # Using Eq. 10 from supp material (cf pdf)
                        p_wt = (self.n_obs_wtm_[m_di][w_di, rt_di] + self.eta) / (self.n_obs_m_[m_di] + self.n_w * self.n_t * self.eta)  # Assumes uniform prior
                        # Using Eq. 11 from supp material (cf pdf)
                        p_o = self._n_obs_do(d, o) / (n_obs_d - 1 + self.alpha)
                        probas[o] = p_wt * p_o
                # New occurrence case
                probas_motif = numpy.zeros((self.n_motifs + 1, ))
                denom_gamma = self.n_occ_ + self.gamma  # Denom in Eq 15
                p_o = self.alpha / (n_obs_d - 1 + self.alpha)
                for m in range(self.n_motifs):
                    p_wt = (self.n_obs_wm_(w_di, m) + self.eta) / (self.n_obs_m_[m] + self.eta * self.n_w * self.n_t)  # Eq 16
                    p_k = self.n_occ_m_[m] / denom_gamma  # Eq 15
                    probas_motif[m] = p_k * p_wt / n_ts_d  # Eq 14
                # New occurrence, new motif case
                probas_motif[-1] = self.gamma / (denom_gamma * self.n_words * n_ts_d)  # p_o factor is given afterwards
                probas[-1] = p_o * numpy.sum(probas_motif)

                draw = numpy.random.multinomial(1, pvals=probas / numpy.sum(probas))
                new_wo_di = numpy.argmax(draw)
                if new_wo_di == n_occ_d:  # New occurrence drawn
                    self.ost[d].append(t_di - numpy.random.randint(self.motif_length))
                    draw = numpy.random.multinomial(1, pvals=probas_motif / numpy.sum(probas_motif))
                    m = numpy.argmax(draw)
                    self.om[d].append(m)
                    self.n_occ_ += 1
                    if m < self.n_motifs:
                        self.n_occ_m_[m] += 1
                    else:
                        self.n_occ_m_.append(1)  # Creating a new motif with a single occurrence
                self._change_occurrence(d, i, old_wo_di, new_wo_di)

    def n_obs_wm_(self, w, m):
        return numpy.sum(self.n_obs_wtm_[m][w, :])

    def _n_obs_do(self, d, o):
        return numpy.sum(self.wo[d] == o)

    def _cancel_obs(self, d, i):
        o = self.wo[d][i]
        if o >= 0:
            m = self.om[d][o]
            self.n_obs_m_[m] -= 1
            rt = self.docs_[d][i, 1] - self.ost[d][o]
            w = self.docs_[d][i, 0]
            self.n_obs_wtm_[m][w, rt] -= 1
        self.wo[d][i] = -1

    def _change_occurrence(self, d, i, old_wo_di, new_wo_di):
        self.wo[d][i] = new_wo_di
        # Update n_obs (only update for new affectation as previous one was already removed in _cancel_obs)
        self._update_n_obs(d, i, new_wo_di)

        if old_wo_di == new_wo_di:
            return None
        if old_wo_di >= 0:
            old_m = self.om[d][old_wo_di]
        else:
            old_m = None

        # Occurrence remapping
        if old_wo_di >= 0 and not self._exists_occurrence(d, old_wo_di):
            self.n_occ_ -= 1
            self.n_occ_m_[old_m] -= 1
            del self.om[d][old_wo_di]
            del self.ost[d][old_wo_di]
            for _i in range(self.wo[d].shape[0]):
                if self.wo[d][_i] > old_wo_di:
                    self.wo[d][_i] -= 1
        # TODO: motif remapping

    def _update_n_obs(self, d, i, o):
        m = self.om[d][o]
        rt = self.docs_[d][i, 1] - self.ost[d][o]
        w = self.docs_[d][i, 0]
        if m < len(self.n_obs_m_):
            self.n_obs_m_[m] += 1
            self.n_obs_wtm_[m][w, rt] += 1
        else:
            self.n_obs_m_.append(1)
            n_obs = numpy.zeros((self.n_words, self.motif_length), dtype=numpy.int)
            n_obs[w, rt] = 1
            self.n_obs_wtm_.append(n_obs)

    def _exists_occurrence(self, d, o):
        for _o in self.wo[d]:
            if _o == o:
                return True
        return False

    def __init_from_data(self, docs):
        self.wo = [numpy.zeros((doc_d.shape[0], ), dtype=numpy.int) - 1 for doc_d in docs]
        self.om = [[] for doc_d in docs]
        self.ost = [[] for doc_d in docs]
        self.docs_ = docs

        self.n_occ_ = 0
        self.n_occ_m_ = []

        self.n_obs_m_ = []
        self.n_obs_wtm_ = []