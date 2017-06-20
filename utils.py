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


def load_trajectories(path, target_type=None):
    """Load trajectories as separate documents.

    Each generated document is a dictionary having two keys:
    - "SHIP_TYPE" gives information about the type of the ship associated to the trajectory
    - "TRAJECTORY" is a list of dictionnaries providing, for each AIS position in the trajectory, longitude, latitude
      and UNIX timestamp.

    Parameters
    ----------
    path : str
        preffix of shapefile names (e.g. "Ascension_032017/Ascension_2017_month3_AIS") from which two shapefiles will
        be considered: path + "_pos" and path + "_track".
    target_type : str or None, default: None
        Ship type to restrict the selection to. If None, all ship types are considered.
    """
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
        if target_type is None or target_type == shprec.record[idx_ship_type]:
            docs.append({"SHIP_TYPE": shprec.record[idx_ship_type],
                         "TRAJECTORY": [{"lon": lon, "lat": lat, "t": pos[idx_timestamp]}
                                        for (lon, lat), pos in zip(shprec.shape.points,
                                                                   positions[idx_start:idx_start + sz_traj])]})
        idx_start += sz_traj
    return docs


def traj_bbox(docs_ais):
    min_lon, max_lon, min_lat, max_lat, min_t, max_t = [numpy.inf, -numpy.inf] * 3
    for d in docs_ais:
        for p in d["TRAJECTORY"]:
            min_lon = min(min_lon, p["lon"])
            max_lon = max(max_lon, p["lon"])
            min_lat = min(min_lat, p["lat"])
            max_lat = max(max_lat, p["lat"])
            min_t = min(min_t, p["t"])
            max_t = max(max_t, p["t"])
    return min_lon, max_lon, min_lat, max_lat, min_t, max_t


def traj_quantize(docs_ais, n_words_lon=210, n_words_lat=200, time_step=10):
    docs_out = []
    min_lon, max_lon, min_lat, max_lat, min_t, max_t = traj_bbox(docs_ais)
    spread_lon = max_lon - min_lon
    spread_lat = max_lat - min_lat
    eps = 1e-9  # For max_lat (resp. max_lon) to fall in bin n_words_lat - 1 (resp. n_words_lon - 1)
    for d in docs_ais:
        d_out = {"SHIP_TYPE": d["SHIP_TYPE"], "TRAJECTORY": []}
        for p in d["TRAJECTORY"]:
            lon_q = int((p["lon"] - min_lon - eps) * n_words_lon / spread_lon)
            lat_q = int((p["lat"] - min_lat - eps) * n_words_lat / spread_lat)
            d_out["TRAJECTORY"].append({"w": lon_q + n_words_lon * lat_q,
                                        "t": (p["t"] - min_t) // time_step})
        docs_out.append(d_out)
    return docs_out


def mix_trajectories(quantized_docs_ais):
    n_obs = sum([len(d["TRAJECTORY"]) for d in quantized_docs_ais])
    final_doc = numpy.empty((n_obs, 2), dtype=numpy.int)
    i = 0
    for d in quantized_docs_ais:
        for p in d["TRAJECTORY"]:
            final_doc[i, 0] = p["w"]
            final_doc[i, 1] = p["t"]
            i += 1
    return final_doc


if __name__ == "__main__":
    doc, n_words, n_t = read_doc_from_json("sample_data/simple_doc.json")
    print(doc)
    print(n_words, n_t)

    print("---")

    docs_ais = load_trajectories("sample_data/Ascension_032017/Ascension_2017_month3_AIS", target_type="Fishing")
    print(set([d["SHIP_TYPE"] for d in docs_ais]))
    print(len(docs_ais))
    print(traj_bbox(docs_ais))
    qdocs = traj_quantize(docs_ais)
    print(mix_trajectories(qdocs).shape)