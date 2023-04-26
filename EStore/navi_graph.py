import rdflib.term
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON
from node import Node
import torch
from embedding_formalism.embed_sot.pykeen_embeddings import pykeen_embeddings
from embedding_formalism.embed_sot.python_rdf2vec import python_rdf2vec
from embedding import InitialEmbedding, DynamicEmbedding
from data_value import DataValue
import uuid
from update_log import InitialUpdate, DynamicUpdate
from embedding_formalism.navipy.manager import get_manager


class NaviGraph:

    def __init__(self) -> None:

        self.nodes = None
        self.nodes_dict = None

        self.relations_object = None
        self.relations_data = None

        self.adjacencies = None
        self.dimension = 0

        self.data_values = dict()
        self.degrees = None

        self.update_logs = list()
        self.embeddings = list()

    def activate(self, link='http://localhost:2020/ds'):

        update = InitialUpdate(uuid.uuid4())

        self.link_query = link
        self.link_update = f'{link}/update'

        sparql_query = SPARQLWrapper(self.link_query)

        sparql_query.setQuery("""
            SELECT distinct ?subject ?predicate ?object WHERE { 
            ?s ?p ?o.
            BIND(
            COALESCE(
            IF(ISBLANK(?s), IRI(?s), ?s)
            ) AS ?subject
            )
            BIND(
            COALESCE(
            IF(ISBLANK(?p), IRI(?p), ?p)
            ) AS ?predicate
            )
            BIND(
            COALESCE(
            IF(ISBLANK(?o), IRI(?o), ?o)
            ) AS ?object
            )
            } 
        """)

        sparql_query.setReturnFormat(JSON)
        results = sparql_query.query().convert()["results"]["bindings"]

        self.nodes = set()
        edges_object = dict()
        edges_data = dict()

        while len(results) > 0:

            popped_result = results.pop(0)

            s_pre, p_pre, o_pre = popped_result['subject'], popped_result['predicate'], popped_result['object']

            s = Node(s_pre['value'])
            self.nodes.add(s)

            if o_pre['type'] == 'uri' or o_pre['type'] == 'bnode':

                o = Node(o_pre['value'])
                self.nodes.add(o)

                try:
                    edges_object[p_pre['value']].append((s, o))
                except KeyError:
                    edges_object[p_pre['value']] = [(s, o)]

            else:
                try:
                    o = DataValue(o_pre['datatype'], o_pre['value'])
                except KeyError:
                    o = DataValue('http://www.w3.org/2001/XMLSchema#string', o_pre['value'])

                try:
                    edges_data[p_pre['value']].append((s, o))
                except KeyError:
                    edges_data[p_pre['value']] = [(s, o)]

        self.nodes = list(self.nodes)
        self.nodes_dict = {node: i for i, node in enumerate(self.nodes)}

        self.relations_object = list(edges_object.keys())
        self.relations_data = list(edges_data.keys())

        self.dimension = len(self.nodes)

        adjacencies_tmp = []

        for rel in self.relations_object:

            size = len(edges_object[rel])
            edges = np.empty((size, 2), dtype=np.int32)

            index = 0
            for s, o in edges_object[rel]:
                if self.nodes_dict[s] > self.dimension or self.nodes_dict[o] > self.dimension:
                    print(s, o, self.nodes_dict[s], self.nodes_dict[o])

                edges[index] = np.array([self.nodes_dict[s], self.nodes_dict[o]])
                index += 1

            print("{} edges added".format(size), rel)

            row, col = np.transpose(edges)

            data = np.ones(len(row), dtype=np.int8)

            adj_shape = (self.dimension, self.dimension)

            i = torch.IntTensor(np.vstack((row, col)))
            v = torch.ByteTensor(data)

            adj_tensor = torch.sparse_coo_tensor(i, v, torch.Size(adj_shape))

            adjacencies_tmp.append(adj_tensor)

        self.adjacencies = torch.stack(adjacencies_tmp)

        for rel in self.relations_data:

            data = dict()

            for s, o in edges_data[rel]:

                try:
                    data[s].append(o)
                except KeyError:
                    data[s] = [o]

            self.data_values[rel] = data

        self.degrees = self.get_degrees()
        update.finish_update(self.adjacencies, self.data_values)
        self.update_logs.append(update)

    def get_degrees(self, adjacencies=None, data_values=None):

        if adjacencies is None:
            adjacencies = self.adjacencies

        if data_values is None:
            data_values = self.data_values

        if adjacencies._nnz() == 0:

            shape = (len(self.relations_object), self.dimension)
            degs_structure_in = torch.sparse_coo_tensor(torch.tensor([[], []]), torch.tensor([]), torch.Size(shape))
            degs_structure_out = torch.sparse_coo_tensor(torch.tensor([[], []]), torch.tensor([]), torch.Size(shape))
            shape = (self.dimension,)
            degs_structure_in_full = torch.sparse_coo_tensor(torch.tensor([[]]), torch.tensor([]), torch.Size(shape)).to_dense().cpu().detach().numpy()
            degs_structure_out_full = torch.sparse_coo_tensor(torch.tensor([[]]), torch.tensor([]), torch.Size(shape)).to_dense().cpu().detach().numpy()

        else:

            degs_structure_in = torch.sparse.sum(adjacencies, dim=1, dtype=torch.long)
            degs_structure_out = torch.sparse.sum(adjacencies, dim=2, dtype=torch.long)
            degs_structure_in_full = torch.sparse.sum(degs_structure_in, dim=0, dtype=torch.long).to_dense().cpu().detach().numpy()
            degs_structure_out_full = torch.sparse.sum(degs_structure_out, dim=0, dtype=torch.long).to_dense().cpu().detach().numpy()

        degs_structure_full = degs_structure_in_full + degs_structure_out_full

        degs_data = np.zeros((len(self.relations_data), len(self.nodes)), dtype=np.int64)

        for i, rel in enumerate(self.relations_data):
            for j, node in enumerate(self.nodes):
                try:
                    degs_data[i, j] += len(data_values[rel][node])
                except:
                    pass

        degs_data_full = degs_data.sum(0)

        degs_full = degs_structure_full + degs_data_full

        return {'structural': {'in': degs_structure_in, 'out': degs_structure_out, 'full': degs_structure_full},
                'data': {'out': degs_data, 'full': degs_data_full},
                'full': degs_full}

    def update(self, results_request):

        update_info = self.get_update_information(results_request)
        print('\n---------------------------------------------------')
        print('DELETED STRUCTURAL:')
        for result in update_info['triples to delete']['structural']:
            print(f'\t{result}')
        print('DELETED DATA:')
        for result in update_info['triples to delete']['data']:
            print(f'\t{result}')

        print('\n---------------------------------------------------')
        print('INSERTED STRUCTURAL:')
        for result in update_info['triples to insert']['structural']:
            print(f'\t{result}')
        print('INSERTED DATA:')
        for result in update_info['triples to insert']['data']:
            print(f'\t{result}')

        self.extend_adjacency_matrix(update_info['new nodes'], update_info['new object relations'])
        adjacencies_delete = self.get_update_matrix(update_info['triples to delete']['structural'])
        adjacencies_insert = self.get_update_matrix(update_info['triples to insert']['structural'])

        self.extend_data_values(update_info['new data relations'])
        data_values_delete = self.get_update_data_values(update_info['triples to delete']['data'])
        data_values_insert = self.get_update_data_values(update_info['triples to insert']['data'])

        # degrees_delete = self.get_degrees(adjacencies_delete, data_values_delete)
        # degrees_insert = self.get_degrees(adjacencies_insert, data_values_insert)

        self.update_adjacency_matrix(adjacencies_insert, adjacencies_delete)
        self.update_data_values(data_values_insert, data_values_delete)

        update = DynamicUpdate(uuid.uuid4(), predecessor_id=self.update_logs[-1].id)
        update.finish_update(adjacencies_delete, data_values_delete, adjacencies_insert, data_values_insert)
        self.update_logs.append(update)

        if len(update_info['new nodes']) > 0:

            self.extend_embedding(update_info['new nodes'])
            array = self.update_embedding(update_info['new nodes'])
            self.assign_embedding(array, initial=False)

            print('\n---------------------------------------------------')
            print('INSERTED EMBEDDINGS FOR:')
            for node in update_info['new nodes']:
                print(f'\t{node.value}')

    def get_update_information(self, results_request):

        triples_delete = results_request['delete']
        triples_insert = results_request['add']

        relevant_triples_delete = {'structural': [], 'data': []}
        relevant_triples_insert = {'structural': [], 'data': []}

        for triple in triples_delete:

            s, p, o = triple['s'], triple['p'], triple['o']
            p = p['value']
            s = Node(s['value'])

            if o['type'] == 'uri' or o['type'] == 'bnode':

                o = Node(o['value'])

                if s in self.nodes and o in self.nodes and p in self.relations_object:
                    if self.adjacencies[self.relations_data.index(p), self.nodes_dict[s], self.nodes_dict[o]] == 1:
                        relevant_triples_delete['structural'].append((s, p, o))

            else:

                try:
                    o = DataValue(o['datatype'], o['value'])
                except KeyError:
                    o = DataValue('http://www.w3.org/2001/XMLSchema#string', o['value'])

                if p in self.relations_data:
                    if s in self.data_values[p].keys():
                        if o in self.data_values[p][s]:
                            relevant_triples_delete['data'].append((s, p, o))

        insert_nodes = set()
        insert_relations_object = set()
        insert_relations_data = set()


        for triple in triples_insert:

            s, p, o = triple['s'], triple['p'], triple['o']

            s = Node(s['value'])
            p = p['value']

            if o['type'] == 'uri' or o['type'] == 'bnode':

                insert_relations_object.add(p)

                o = Node(o['value'])

                insert_nodes.add(s)
                insert_nodes.add(o)

                if s in self.nodes and o in self.nodes and p in self.relations_object:
                    if self.adjacencies[self.relations_object.index(p), self.nodes_dict[s], self.nodes_dict[o]] == 0:
                        relevant_triples_insert['structural'].append((s, p, o))
                else:
                    relevant_triples_insert['structural'].append((s, p, o))

            else:

                insert_relations_data.add(p)

                try:
                    o = DataValue(o['datatype'], o['value'])
                except KeyError:
                    o = DataValue('http://www.w3.org/2001/XMLSchema#string', o['value'])

                if p in self.relations_data:
                    if s in self.data_values[p].keys():
                        if o in self.data_values[p][s]:
                            continue

                relevant_triples_insert['data'].append((s, p, o))

        new_nodes = insert_nodes - set(self.nodes)
        new_relations_object = insert_relations_object - set(self.relations_object)
        new_relations_data = insert_relations_data - set(self.relations_data)

        return {'triples to delete': relevant_triples_delete, 'triples to insert': relevant_triples_insert,
                'new nodes': new_nodes, 'new object relations': new_relations_object,
                'new data relations': new_relations_data}

    def extend_adjacency_matrix(self, new_nodes, new_relations):

        if len(new_nodes) == 0 and len(new_relations) == 0:
            pass
        else:

            for node in new_nodes:
                self.nodes_dict[node] = len(self.nodes)
                self.nodes.append(node)

            for relation in new_relations:
                self.relations_object.append(relation)

            num_relations = len(self.relations_object)
            dimension = len(self.nodes)

            self.adjacencies = torch.sparse_coo_tensor(self.adjacencies._indices(), self.adjacencies._values(),
                                                    torch.Size((num_relations, dimension, dimension)))

            self.dimension = dimension

    def get_update_matrix(self, triples):

        rel = []
        x = []
        y = []

        for s, p, o in triples:

            rel.append(self.relations_object.index(p))
            x.append(self.nodes_dict[s])
            y.append(self.nodes_dict[o])

        indices = np.array([rel, x, y], dtype=np.int32)

        values = np.ones((len(triples),), dtype=np.uint8)

        i = torch.IntTensor(indices)
        v = torch.ByteTensor(values)

        shape = (len(self.relations_object), self.dimension, self.dimension)

        adj_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape))

        return adj_tensor

    def update_adjacency_matrix(self, adjacencies_insert=None, adjacencies_delete=None):

        if adjacencies_insert is not None:

            self.adjacencies += adjacencies_insert

        if adjacencies_delete is not None:
            self.adjacencies -= adjacencies_delete

    def extend_data_values(self, new_relations):

        if len(new_relations) == 0:
            pass
        else:
            for relation in new_relations:
                self.relations_data.append(relation)
                self.data_values[relation] = dict()

    def get_update_data_values(self, triples):

        data_values = dict()

        for relation in self.relations_data:

            data_values[relation] = dict()

        for s, p, o in triples:

            try:
                data_values[p][s].append(o)
            except KeyError:
                data_values[p][s] = [o]

        return data_values

    def update_data_values(self, data_values_insert=None, data_values_delete=None):

        for relation in self.relations_data:

            for subject, value_list in data_values_insert[relation].items():
                try:
                    self.data_values[relation][subject].extend(value_list)
                except KeyError:
                    self.data_values[relation][subject] = value_list

            for subject, value_list in data_values_delete[relation].items():
                self.data_values[relation][subject] = [x for x in self.data_values[relation][subject] if x not in value_list]

    def extend_embedding(self, new_nodes):

        num_new_nodes = len(new_nodes)

        if num_new_nodes > 0:

            dim_embedding = self.embeddings[-1].array.shape[1]
            append = torch.tensor([[float(0)]*dim_embedding]*num_new_nodes)
            self.embeddings[-1].array = torch.cat((self.embeddings[-1].array, append), dim=0)

    def update_embedding(self, new_nodes):

        matrix_insert = self.adjacencies

        if len(new_nodes) == 0:
            return

        new_indices = [self.nodes_dict[node] for node in new_nodes]

        features = self.embeddings[-1].array.clone()

        matrix_insert_t = torch.sparse_coo_tensor(torch.index_select(matrix_insert._indices(), 0, torch.tensor([0, 2, 1])),
                                                matrix_insert._values(), matrix_insert.shape)

        num_new_indices = len(new_indices)
        num_existing_indices = self.dimension - num_new_indices
        num_nodes = len(self.nodes)
        num_relations = len(self.relations_object)
        dimension = matrix_insert.shape[1]

        reducer2rows = torch.sparse_coo_tensor(
            [[x for x in range(num_new_indices)], [x + num_existing_indices for x in range(num_new_indices)]],
            [1] * num_new_indices, torch.Size((num_new_indices, num_nodes))).float()
        reducer2cols = torch.sparse_coo_tensor(
            [[x + num_existing_indices for x in range(num_new_indices)], [x for x in range(num_new_indices)]],
            [1] * num_new_indices, torch.Size((num_nodes, num_new_indices))).float()

        matrix_insert_new_rows = torch.stack(
            [torch.sparse.mm(reducer2rows, matrix_insert[i].float()) for i in range(num_relations)], dim=0)
        matrix_insert_t_new_rows = torch.stack(
            [torch.sparse.mm(reducer2rows, matrix_insert_t[i].float()) for i in range(num_relations)], dim=0)
        matrix_insert_new_cols = torch.stack(
            [torch.sparse.mm(matrix_insert[i].float(), reducer2cols) for i in range(num_relations)], dim=0)
        matrix_insert_t_new_cols = torch.stack(
            [torch.sparse.mm(matrix_insert_t[i].float(), reducer2cols) for i in range(num_relations)], dim=0)

        indices_connect_order = []
        indices_tmp = new_indices.copy()

        while not all(v is None for v in indices_tmp):

            indices_to_keep = list(set(list(range(num_nodes))) - set(indices_tmp))
            mask_matrix = torch.sparse_coo_tensor([indices_to_keep, indices_to_keep], [1] * len(indices_to_keep),
                                                  torch.Size((num_nodes, num_nodes))).float()

            matrix_insert_new_rows_masked = torch.stack(
                [torch.sparse.mm(matrix_insert_new_rows[i], mask_matrix) for i in range(num_relations)], dim=0)
            matrix_insert_t_new_rows_masked = torch.stack(
                [torch.sparse.mm(matrix_insert_t_new_rows[i], mask_matrix) for i in range(num_relations)], dim=0)

            matrix_insert_new_cols_masked = torch.stack(
                [torch.sparse.mm(mask_matrix, matrix_insert_new_cols[i]) for i in range(num_relations)], dim=0)
            matrix_insert_t_new_cols_masked = torch.stack(
                [torch.sparse.mm(mask_matrix, matrix_insert_t_new_cols[i]) for i in range(num_relations)], dim=0)

            matrix_insert_new_cols_masked_t = torch.sparse_coo_tensor(
                torch.index_select(matrix_insert_new_cols_masked._indices(), 0, torch.tensor([0, 2, 1])),
                matrix_insert_new_cols_masked._values(), matrix_insert_new_rows_masked.shape)
            matrix_insert_t_new_cols_masked_t = torch.sparse_coo_tensor(
                torch.index_select(matrix_insert_t_new_cols_masked._indices(), 0, torch.tensor([0, 2, 1])),
                matrix_insert_t_new_cols_masked._values(), matrix_insert_new_rows_masked.shape)

            full_degrees_adj = matrix_insert_t_new_rows_masked + matrix_insert_t_new_rows_masked + matrix_insert_new_cols_masked_t + matrix_insert_t_new_cols_masked_t

            degrees = torch.sparse.sum(torch.sparse.sum(full_degrees_adj, dim=2), dim=0)
            degrees_np = list(degrees.to_dense().cpu().detach().numpy())

            indices2update = [indices_tmp[i] for i in range(len(indices_tmp)) if
                              degrees_np[i] != 0 and indices_tmp[i] is not None]

            if len(indices2update) > 0:
                indices_connect_order.append(indices2update)
                for i in indices2update:
                    indices_tmp[indices_tmp.index(i)] = None
            else:
                indices_connect_order.append(indices_tmp)
                indices_tmp = [None]*len(indices_tmp)

        for indices in indices_connect_order:

            features_small = self.reconstructor['trained'].reconstruct_embeddings(self, indices=indices)

            for i, index in enumerate(indices):

                features[index] = features_small[i]

        return features

    def embed_sot(self, ground_setting):

        rdflibgraph = self.navigraph2rdflib(literal=False)

        embedding_engine = ground_setting['embedding engine']

        if embedding_engine == 'pykeen':
            model_str = ground_setting['model string']
            epochs = int(ground_setting['epochs'])
            dim = int(ground_setting['dimension'])
            print('\tGenerating', model_str, 'embedding with', epochs, 'epochs.')
            array = pykeen_embeddings(rdflibgraph, self.nodes, model_str, epochs, dim)

        if embedding_engine == 'rdf2vec':
            epochs = int(ground_setting['epochs'])
            dim = int(ground_setting['dimension'])
            depth = int(ground_setting['depth'])
            number_walks = int(ground_setting['number of walks'])
            print('\tGenerating RDF2Vec embedding with: epochs: '
                  + str(epochs) + ', depth: ' + str(depth) + ', # of walks: ' + str(number_walks))
            array = python_rdf2vec(rdflibgraph, self.nodes, epochs, depth, number_walks, dim)

        self.assign_embedding(array, ground_setting)

    def assign_embedding(self, array, ground_setting=None, initial=True):

        update_id = self.update_logs[-1].id
        if len(self.embeddings) > 0:
            precedent_id = self.embeddings[-1].id
        else:
            precedent_id = None

        if initial:
            self.embeddings.append(InitialEmbedding(array, self.nodes, update_id, precedent_id=precedent_id,
                                            setting=ground_setting))
        else:
            self.embeddings.append(DynamicEmbedding(array, self.nodes, update_id, precedent_id=precedent_id,
                                                    setting=self.reconstructor_args))

    def assign_reconstructor(self, reconstructor_args):
        self.reconstructor = {'initial': get_manager(reconstructor_args, True)}
        self.reconstructor_args = reconstructor_args

    def train_reconstructor(self):

        assert hasattr(self, 'reconstructor')
        assert hasattr(self, 'embeddings')
        manager = get_manager(self.reconstructor_args, False)
        manager.do_train(self.embeddings[-1].array, self.adjacencies)
        self.reconstructor['trained'] = manager

    def navigraph2rdflib(self, uri=True, literal=True):

        graph = rdflib.Graph()

        if uri:
            for relation in self.relations_object:
                indices = self.adjacencies[self.relations_object.index(relation)]._indices().cpu().detach().numpy()
                indices = list(zip(list(indices[0]), list(indices[1])))
                for x, y in indices:
                    graph.add((self.nodes[x].node2rdflib(), rdflib.term.URIRef(relation), self.nodes[y].node2rdflib()))
        if literal:
            for relation in self.relations_data:
                for subject, values in self.data_values[relation].items():
                    for value in values:
                        graph.add((subject.node2rdflib(), rdflib.term.URIRef(relation), value.literal2rdflib()))

        return graph
