import json
import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def read_doc_from_json(fname):
    doc_dict = json.load(open(fname, "rt"))
    n_obs = len(doc_dict)
    npy_doc = numpy.empty((n_obs, 2), dtype=numpy.int)
    for i in range(n_obs):
        npy_doc[i, :] = [doc_dict[i]["w"], doc_dict[i]["t"]]
    n_words = numpy.max(npy_doc[:, 0]) + 1
    n_t = numpy.max(npy_doc[:, 1]) + 1
    return npy_doc, n_words, n_t


if __name__ == "__main__":
    doc, n_words, n_t = read_doc_from_json("sample_data/simple_doc.json")
    print(doc)
    print(n_words, n_t)