from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker
import numpy as np
import multiprocessing
import os
import torch


def python_rdf2vec(rdflibgraph, nodes, epochs, depth, number_walks, dim):

    dir = os.path.dirname(__file__)
    tmp_file = f'{dir}/tmp/rdf2vec_tmp.ttl'

    rdflibgraph.serialize(destination=tmp_file)

    knowledge_graph = KG(tmp_file)

    nodes = [node.value for node in nodes]

    transformer = RDF2VecTransformer(
        Word2Vec(epochs=epochs),
        walkers=[RandomWalker(depth, number_walks, with_reverse=False, n_jobs=multiprocessing.cpu_count() // 2)],
        verbose=1
    )
    # Get our embeddings.
    embeddings, literals = transformer.fit_transform(knowledge_graph, nodes)
    embeddings_np = np.array(embeddings)
    embeddings = torch.from_numpy(embeddings_np)

    os.remove(tmp_file)

    return embeddings
