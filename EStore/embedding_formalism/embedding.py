import numpy as np
from scipy.sparse import csr_matrix
from kg.embed_sot.pykeen_embeddings import pykeen_embeddings
from kg.embed_sot.python_rdf2vec import python_rdf2vec


def embed_navi(skg_1, skg_2, reconstructor, initial, operation='insert', alpha=None):

    assert skg_1.embedding is not None
    if initial:
        manager = reconstructor['initial']
    else:
        manager = reconstructor['trained']

    degrees_in = skg_1.get_degrees()
    degrees_out = skg_2.get_degrees()

    if operation == 'insert':

        urirefs_1 = skg_1.urirefs
        urirefs_2 = skg_2.urirefs

        uris_new = list(set(urirefs_2) - set(urirefs_1))

        if uris_new:

            uris_existing = list(set(urirefs_1).intersection(set(urirefs_2)))

            uris_new_indices =[]
            for uri in uris_new:
                uris_new_indices.append(skg_2.urirefs_dict[uri])

            uris_existing_indices = []
            for uri in uris_existing:
                uris_existing_indices.append(skg_2.urirefs_dict[uri])

            reduced_adjacencies = {}
            new_degrees = np.array([0]*len(uris_new))

            for relation in skg_2.object_relations:

                a0 = skg_2.adjacency_matrices[relation][uris_new_indices, :][:, uris_existing_indices]
                new_degrees += np.asarray(a0.sum(axis=1).transpose())[0]
                a1 = skg_2.adjacency_matrices[relation].transpose()[uris_new_indices, :][:, uris_existing_indices]
                new_degrees += np.asarray(a1.sum(axis=1).transpose())[0]
                reduced_adjacencies[relation] = (a0, a1)

            row_switch_row = []

            uris_2_without_new_nodes = []
            for uri in skg_2.urirefs:
                if uri in uris_existing:
                    uris_2_without_new_nodes.append(uri)
            uris_2_without_new_nodes = {node: i for i, node in enumerate(uris_2_without_new_nodes)}

            for node in skg_1.urirefs:
                row_switch_row.append(uris_2_without_new_nodes[node])

            row_switch_row = np.array(row_switch_row)
            row_switch_col = np.array(list(range(len(skg_1.urirefs))))
            row_switch_data = np.array([1]*len(skg_1.urirefs))
            row_switch_matrix = csr_matrix((row_switch_data, (row_switch_row, row_switch_col)),
                                           shape=(len(skg_1.urirefs), len(skg_1.urirefs)))

            new_embeddings = manager.create_new_embeddings(skg_1, row_switch_matrix, reduced_adjacencies)

        embedding_tmp = [None]*len(skg_2.urirefs)
        degrees_tmp = [None] * len(skg_2.urirefs)

        for i, uri in enumerate(skg_1.urirefs):
            embedding_tmp[skg_2.urirefs_dict[uri]] = skg_1.embedding[i]
            degrees_tmp[skg_2.urirefs_dict[uri]] = degrees_in[i]

        if uris_new:
            for i, uri in enumerate(uris_new):
                embedding_tmp[skg_2.urirefs_dict[uri]] = new_embeddings[i]
                degrees_tmp[skg_2.urirefs_dict[uri]] = new_degrees[i]

        degrees_in = degrees_tmp

        embedding = manager.reconstruct_embeddings(skg_2, np.array(embedding_tmp))
        alpha = np.array([1-d_in/d_out for d_in, d_out in zip(degrees_in, degrees_out)]) if alpha is None else alpha
        alpha_inv = 1 - alpha
        return np.multiply(embedding, alpha[:, np.newaxis]) + np.multiply(embedding_tmp, alpha_inv[:, np.newaxis])


    elif operation == 'delete':

        embedding_tmp = [None] * len(skg_2.urirefs)
        degrees_tmp = [None] * len(skg_2.urirefs)

        for i, uri in enumerate(skg_2.urirefs):
            embedding_tmp[i] = skg_1.embedding[skg_1.urirefs_dict[uri]]
            degrees_tmp[i] = degrees_in[skg_1.urirefs_dict[uri]]

        degrees_in = degrees_tmp
        embedding = manager.reconstruct_embeddings(skg_2, np.array(embedding_tmp))
        alpha = np.array([1-d_out/d_in for d_in, d_out in zip(degrees_in, degrees_out)]) if alpha is None else alpha
        alpha_inv = 1 - alpha
        return np.multiply(embedding, alpha[:, np.newaxis]) + np.multiply(embedding_tmp, alpha_inv[:, np.newaxis])

    else:
        raise Exception('I know Python!')


def regularize_navi(navigraph, initial, alpha=0.5):

    if initial:
        manager = navigraph.reconstructor['initial']
    else:
        manager = navigraph.reconstructor['trained']

    embedding = manager.reconstruct_embeddings(navigraph)

    return alpha * embedding + (1-alpha) * navigraph.embeddings[-1].array


def embed_sot(skg, meta):

    embedding_engine = meta['embedding engine']

    if embedding_engine == 'pykeen':
        model_str = meta['model string']
        epochs = meta['epochs']
        dim = meta['dimension']
        print('\tGenerating', model_str, 'embedding with', epochs, 'epochs.')
        embedding = pykeen_embeddings(skg, model_str, epochs, dim)

    if embedding_engine == 'rdf2vec':
        epochs = meta['epochs']
        dim = meta['dimension']
        depth = meta['depth']
        number_walks = meta['number of walks']
        print('\tGenerating RDF2Vec embedding with: epochs: '
              + str(epochs) + ', depth: ' + str(depth) + ', # of walks: ' + str(number_walks))
        embedding = python_rdf2vec(skg, epochs, depth, number_walks, dim)

    return embedding

