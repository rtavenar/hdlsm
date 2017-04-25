import numpy
import pylab

from hdlsm_model import HierarchicalDirichletLatentSemanticMotifs as HDLSM
from utils import read_doc_from_json

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

numpy.random.seed(0)

doc, n_words, n_t = read_doc_from_json("sample_data/simple_doc.json")
model = HDLSM(motif_length=5, n_words=n_words, alpha=0.1, eta=0.1, gamma=0.1)
model.fit([doc])
print(model.wo[0])

for m in range(model.n_motifs()):
    n_occ = model.n_occ_m_[m]
    print("Motif %d: %d occurrences" % (m, n_occ))
    if n_occ > 0:
        motif = model.p_wt_m(m)

        pylab.figure(figsize=(model.motif_length, model.n_words))
        pylab.imshow(motif, vmin=0., interpolation="none", cmap="gray_r")
        pylab.ylabel("Word index")
        pylab.xlabel("Relative time")
        pylab.savefig("output/simple_doc_motif%d.png" % m)
