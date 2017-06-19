import json
import numpy
import shapefile

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


def load_trajectories(path):
    path_pos = path + "_pos"
    path_track = path + "_track"

    sf_track = shapefile.Reader(path_track)
    idx_ship_type = 4
    assert sf_track.fields[idx_ship_type + 1][0] == "SHIP_TYPE"

    sf_pos = shapefile.Reader(path_pos)
    idx_timestamp = 14
    assert sf_pos.fields[idx_timestamp + 1][0] == "UNIX_TIME"
    positions = sf_pos.records()

    docs = []
    idx_start = 0
    for idx_shape, shprec in enumerate(sf_track.shapeRecords()):
        sz_traj = len(shprec.shape.points)
        docs.append({"SHIP_TYPE": shprec.record[idx_ship_type],
                     "TRAJECTORY": [{"lon": lon, "lat": lat, "t": pos[idx_timestamp]}
                                    for (lon, lat), pos in zip(shprec.shape.points,
                                                               positions[idx_start:idx_start+sz_traj])]})
        idx_start += sz_traj
    return docs


if __name__ == "__main__":
    doc, n_words, n_t = read_doc_from_json("sample_data/simple_doc.json")
    print(doc)
    print(n_words, n_t)

    print("---")

    docs_ais = load_trajectories("sample_data/Ascension_032017/Ascension_2017_month3_AIS")
    print(len(docs_ais))

