from pykeen.pipeline import pipeline
import os
import numpy as np
import torch
from embedding_formalism.embed_sot.create_tsv import create_tsv


def get_embedding(rdflib_graph, nodes, tmp_file_full, tmp_file_test, model_str, epochs, dim):

    create_tsv(rdflib_graph, tmp_file_full, tmp_file_test)

    result = pipeline(
        model=model_str,
        training=os.path.abspath(tmp_file_full),
        testing=os.path.abspath(tmp_file_test),
        epochs=epochs,
        model_kwargs=dict(embedding_dim=dim)
    )

    model = result.model

    entity_embeddings = model.entity_representations[0](indices=None).cpu().detach().numpy()

    entity_to_id_dict = result.training.entity_to_id

    word_vectors = {}

    for k, v in entity_to_id_dict.items():
        word_vectors[k] = entity_embeddings[v]

    nodes = [node.value for node in nodes]

    embeddings = []
    for node in nodes:
        embeddings.append(word_vectors[node])

    embeddings_np = np.array(embeddings)
    embeddings = torch.from_numpy(embeddings_np)

    os.remove(tmp_file_full)
    os.remove(tmp_file_test)

    return embeddings


def pykeen_embeddings(rdflibgraph, nodes, model_str='TransE', epochs=100, dim=100):

    dir = os.path.dirname(__file__)

    tmp_file_full = f'{dir}/tmp/tmp.tsv'
    tmp_file_test = f'{dir}/tmp/tmp_test.tsv'

    try:
        os.remove(tmp_file_full)
        os.remove(tmp_file_test)
    except OSError:
        pass

    embeddings_np = get_embedding(rdflibgraph, nodes, tmp_file_full, tmp_file_test, model_str, epochs, dim)

    return embeddings_np
