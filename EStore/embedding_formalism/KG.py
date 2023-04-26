import rdflib
import gzip
from collections import Counter
import numpy as np
from scipy.sparse import coo_matrix
import pickle, os
from abc import ABC, abstractmethod
from kg.navipy.manager import get_manager


class KG(ABC):

    def __init__(self):
        super(KG, self).__init__()
        self.activate()

    def serialize(self, filename):

        with open(f'{filename}.pkl', 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{filename}.pkl', 'rb') as handle:
            with gzip.open(f'{filename}.pkl.gz', 'wb') as zipped_file:
                zipped_file.writelines(handle)
        os.remove(f'{filename}.pkl')


class StationaryKG(KG, ABC):

    def __init__(self, timestamp=None, embedding=None):
        super(StationaryKG, self).__init__()
        self.timestamp = timestamp
        if embedding is not None:
            self.assign_embedding(embedding, True)
        else:
            self.embedding = None

    @abstractmethod
    def activate(self):
        pass

    @abstractmethod
    def check_embedding(self):
        pass

    @abstractmethod
    def recreate_kg(self):
        pass

    def assign_embedding(self, embedding, meta, sot_embedding=True):
        self.check_embedding(embedding)
        self.embedding = embedding
        self.embedding_meta = meta
        self.sot_embedding = sot_embedding


class StandardKG(StationaryKG):

    def __init__(self, graph_file_name, identity, timestamp=None):
        self.graph_file_name = graph_file_name
        self.id = identity
        super(StandardKG, self).__init__(timestamp)

    def activate(self):

        g = rdflib.Graph()

        if self.graph_file_name.endswith('.gz'):
            with gzip.open(self.graph_file_name, 'rb') as f:
                g.parse(file=f)
        else:
            g.parse(self.graph_file_name)

        freq = Counter(g.predicates())
        relation_frequencies = freq.most_common()
        self.relations = [x[0] for x in relation_frequencies]

        subjects = set(g.subjects())
        objects = set(g.objects())
        nodes = list(subjects.union(objects))

        self.urirefs = [x for x in nodes if isinstance(x, rdflib.URIRef)]

        blanknodes = [x for x in nodes if isinstance(x, rdflib.BNode)]
        if len(blanknodes) > 0:
            self.blanknodes = True
            self.g_blanknodes = rdflib.Graph()
            for s, p, o in g.triples((None, None, None)):
                if isinstance(s, rdflib.term.BNode) or isinstance(o, rdflib.term.BNode):
                    self.g_blanknodes.add((s, p, o))
        else:
            self.blanknodes = False

        self.object_relations = []
        self.data_relations = []

        for relation in self.relations:

            property_type = list(
                g.triples((relation, rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), None)))

            if len(property_type) == 0:
                self.object_relations.append(relation)

            if len(property_type) == 1:
                if property_type[0][2] == rdflib.term.URIRef('http://www.w3.org/2002/07/owl#ObjectProperty'):
                    self.object_relations.append(relation)
                else:
                    self.data_relations.append(relation)

        self.g_literal = rdflib.Graph()

        for relation in self.data_relations:
            for s, p, o in g.triples((None, relation, None)):
                self.g_literal.add((s, p, o))

        self.adjacency_matrices = {}

        self.urirefs_dict = {uri: i for i, uri in enumerate(self.urirefs)}

        for relation in self.object_relations:

            adjacency_matrix = coo_matrix((len(self.urirefs), len(self.urirefs)), dtype=np.int8)
            adjacency_matrix = adjacency_matrix.todok()

            for s, p, o in g.triples((None, relation, None)):
                if isinstance(s, rdflib.term.BNode) or isinstance(o, rdflib.term.BNode):
                    continue
                adjacency_matrix[self.urirefs_dict[s], self.urirefs_dict[o]] = 1

            self.adjacency_matrices[relation] = adjacency_matrix.tocsr()

    def check_embedding(self, embedding):

        assert len(self.urirefs) == len(embedding)

    def recreate_kg(self, uri=True, bnode=False, literal=False):

        g = rdflib.Graph()

        if uri:
            for relation in self.object_relations:
                x, y = self.adjacency_matrices[relation].nonzero()
                indices = list(zip(list(x), list(y)))
                for x, y in indices:
                    g.add((self.urirefs[x], relation, self.urirefs[y]))
        if bnode:
            for s, p, o in self.g_blanknodes.triples((None, None, None)):
                g.add((s, p, o))
        if literal:
            for s, p, o in self.g_literal.triples((None, None, None)):
                g.add((s, p, o))

        return g

    def get_degrees(self, relation=None):

        if relation is not None:

            col_sum = self.adjacency_matrices[relation].sum(axis=0)
            row_sum = self.adjacency_matrices[relation].sum(axis=1).transpose()
            diag = self.adjacency_matrices[relation].diagonal()

            degrees = np.asarray(col_sum + row_sum - diag)[0]

            return degrees

        else:

            degrees = None

            for relation in self.object_relations:

                col_sum = self.adjacency_matrices[relation].sum(axis=0)
                row_sum = self.adjacency_matrices[relation].sum(axis=1).transpose()
                diag = self.adjacency_matrices[relation].diagonal()

                if degrees is None:
                    degrees = np.asarray(col_sum + row_sum - diag)[0]
                else:
                    degrees += np.asarray(col_sum + row_sum - diag)[0]

            return degrees

    def assign_reconstructor(self, reconstructor_args):
        self.reconstructor = {'initial': get_manager(reconstructor_args, True)}
        self.reconstructor_args = reconstructor_args

    def train_reconstructor(self):

        assert hasattr(self, 'reconstructor')
        assert hasattr(self, 'embedding')
        manager = get_manager(self.reconstructor_args, False)
        manager.do_train(self)
        self.reconstructor['trained'] = manager


class DynamicKG(KG, ABC):

    def __init__(self):
        super(DynamicKG, self).__init__()

    def add_image(self, stationary_kg):
        self.images.append(stationary_kg)

    def activate(self):
        self.images = list()


class MutableKG(DynamicKG):

    def __init__(self, identity):
        self.id = identity
        super(MutableKG, self).__init__()

    def assign_reconstructor(self, reconstructor, kg_id):
        self.reconstructor = reconstructor
        self.reconstructor_id = kg_id


if __name__ == '__main__':

    dynamic_kg = MutableKG('aifb')
    stationary_kg = StandardKG('aifb_stripped.nt.gz', 'aifb-0')
    dynamic_kg.add_image(stationary_kg)
    # stationary_kg = StandardKG('aifb_stripped.nt.gz', 'aifb-1')
    # dynamic_kg.add_image(stationary_kg)
    dynamic_kg.serialize('data')
    print('hi')
    print('hi')
