from hdlsm_model import HierarchicalDirichletLatentSemanticMotifs as HDLSM
from utils import read_doc_json

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

doc, n_words, n_t = read_doc_json("sample_data/simple_doc.json")
m = HDLSM(motif_length=5, n_words=n_words, alpha=0.1, eta=0.1, gamma=0.1)
m.fit([doc])

print(m.wo[0])