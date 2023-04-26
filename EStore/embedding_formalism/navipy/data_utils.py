from __future__ import print_function
import gzip
import numpy as np
import scipy.sparse as sp
import rdflib as rdf
import pickle as pkl
from collections import Counter
import random


class RDFReader:
    __graph = None
    __freq = {}

    def __init__(self, file):

        self.__graph = rdf.Graph()

        if file.endswith(".gz"):
            with gzip.open(file, "rb") as f:
                self.__graph.parse(file=f)
        else:
            self.__graph.parse(file, format=rdf.util.guess_format(file))

        # See http://rdflib.readthedocs.io for the rdflib documentation

        self.__freq = Counter(self.__graph.predicates())

        print("Graph loaded, frequencies counted.")

    def triples(self, relation=None):
        for s, p, o in self.__graph.triples((None, relation, None)):
            yield s, p, o

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__graph.destroy("store")
        self.__graph.close(True)

    def subjectSet(self):
        return set(self.__graph.subjects())

    def objectSet(self):
        return set(self.__graph.objects())

    def relationList(self):
        """
        Returns a list of relations, ordered descending by frequenecy
        :return:
        """
        res = list(set(self.__graph.predicates()))
        res.sort(key=lambda rel: -self.freq(rel))
        return res

    def __len__(self):
        return len(self.__graph)

    def freq(self, relation):
        """zzz
        The frequency of this relation (how many distinct triples does it occur in?)
        :param relation:
        :return:
        """
        if relation not in self.__freq:
            return 0
        return self.__freq[relation]


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix(
        (loader["data"], loader["indices"], loader["indptr"]),
        shape=loader["shape"],
        dtype=np.float32,
    )


def get_features(dataset_str="aifb", embedding_engine=None, embedding_file=None, split_type='meta'):

    base = f'Data/{dataset_str}/embeddings/{embedding_engine}/'

    features = np.load(base + embedding_file + f'_{split_type}.npy')

    return features


def generate_adjacencies_full(dataset_str="aifb", split_type='meta'):

    base = f'Data/{dataset_str}/{split_type}/'

    print("Loading dataset", dataset_str)

    graph_file = base + 'uri.ttl.gz'

    with RDFReader(graph_file) as reader:

        relations = reader.relationList()
        print([(rel, reader.freq(rel)) for rel in relations])

        with open(base + 'nodes.pkl', 'rb') as f:
            nodes = pkl.load(f)

        adj_shape = (len(nodes), len(nodes))

        print("Number of nodes: ", len(nodes))
        print("Number of relations in the data: ", len(relations))

        relations_dict = {rel: i for i, rel in enumerate(list(relations))}
        nodes_dict = {node: i for i, node in enumerate(nodes)}

        adjacencies = []

        for i, rel in enumerate(relations):

            print(
                u"Creating adjacency matrix for relation {}: {}, frequency {}".format(
                    i, rel, reader.freq(rel)
                )
            )

            size = 0
            for s, p, o in reader.triples(relation=rel):
                size += 1

            edges = np.empty((size, 2), dtype=np.int32)

            index = 0
            for s, p, o in reader.triples(relation=rel):
                s, p, o = s.toPython(), p.toPython(), o.toPython()
                if nodes_dict[s] > len(nodes) or nodes_dict[o] > len(nodes):
                    print(s, o, nodes_dict[s], nodes_dict[o])

                edges[index] = np.array([nodes_dict[s], nodes_dict[o]])
                index += 1

            print("{} edges added".format(size))

            row, col = np.transpose(edges)

            data = np.ones(len(row), dtype=np.int8)

            adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.int8)

            adj_transp = sp.csr_matrix(
                (data, (col, row)), shape=adj_shape, dtype=np.int8
            )

            adjacencies.append(adj)
            adjacencies.append(adj_transp)

    return adjacencies, relations_dict


def generate_adjacencies_light(dataset_str="aifb", rel_dict=dict(), targets_split = dict(), split_type=None):

    """
    :param dataset_str:
    :param rel_layers:
    :param limit: If > 0, will only load this many adj. matrices
        All adjacencies are preloaded and saved to disk,
        but only a limited a then restored to memory.
    :return:
    """

    groundbase = f'Data/{dataset_str}/'
    base = f'Data/{dataset_str}/{split_type}/'

    print("Loading dataset", dataset_str)

    graph_file = groundbase + 'uri.ttl.gz'

    targets_full = targets_split['full']
    targets_navi = targets_split['navi']

    print(f'Loading dataset {dataset_str} for split type {split_type}')

    with RDFReader(graph_file) as reader:

        with open(base + 'nodes.pkl', 'rb') as f:
            nodes = pkl.load(f)

        print("Number of nodes: ", len(nodes))

        nodes_dict = {node: i for i, node in enumerate(nodes)}

        targets_full_dict = {target: i for i, target in enumerate(targets_full)}

        adjacencies = []

        test_nodes = set()

        for i, rel in enumerate(rel_dict.keys()):

            size_out = 0
            size_in = 0

            for s, p, o in reader.triples(relation=rel):

                s, p, o = s.toPython(), p.toPython(), o.toPython()

                if s in targets_full and o in targets_full:
                    if s in targets_navi or o in targets_navi:
                        continue
                if o in targets_full:
                    try:
                        nodes_dict[s]
                        size_in += 1
                    except KeyError:
                        continue

            for s, p, o in reader.triples(relation=rel):

                s, p, o = s.toPython(), p.toPython(), o.toPython()

                if s in targets_full and o in targets_full:
                    if s in targets_navi or o in targets_navi:
                        continue
                if s in targets_full:
                    try:
                        nodes_dict[o]
                        size_out += 1
                    except KeyError:
                        continue

            edges_out = np.empty((size_out, 2), dtype=np.int32)
            edges_in = np.empty((size_in, 2), dtype=np.int32)

            index = 0
            for s, p, o in reader.triples(relation=rel):

                s, p, o = s.toPython(), p.toPython(), o.toPython()

                if s in targets_full and o in targets_full:
                    if s in targets_navi or o in targets_navi:
                        continue
                if o in targets_full:
                    try:
                        edges_in[index] = np.array([nodes_dict[s], targets_full_dict[o]])
                        index += 1
                    except KeyError:
                        print('in', s,p,o)


            index = 0
            for s, p, o in reader.triples(relation=rel):

                s, p, o = s.toPython(), p.toPython(), o.toPython()

                if s in targets_full and o in targets_full:
                    if s in targets_navi or o in targets_navi:
                        continue
                if s in targets_full:
                    try:
                        edges_out[index] = np.array([nodes_dict[o], targets_full_dict[s]])
                        index += 1
                    except KeyError:
                        print('out', s,p,o)

            row, col = np.transpose(edges_in)
            data = np.ones(len(row), dtype=np.int8)
            adj = sp.csr_matrix((data, (row, col)), shape=(len(nodes), len(targets_full)), dtype=np.int8)

            row, col = np.transpose(edges_out)
            data = np.ones(len(row), dtype=np.int8)
            adj_transp = sp.csr_matrix((data, (row, col)), shape=(len(nodes), len(targets_full)), dtype=np.int8)

            adjacencies.append(adj)
            adjacencies.append(adj_transp)

    return adjacencies


def split_data(nodes, train_portion=0.9):

    idx = list(range(len(nodes)))

    random.shuffle(idx)

    length_train = int(len(idx)*train_portion)

    return idx[:length_train], idx[length_train:]
