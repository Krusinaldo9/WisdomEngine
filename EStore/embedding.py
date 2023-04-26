import numpy as np
import uuid
import datetime
from abc import ABC


class Embedding(ABC):

    def __init__(self, array: np.float32, nodes: list(), update_id: uuid.UUID, precedent_id: uuid.UUID = None,
                 setting=None):
        super(Embedding, self).__init__()

        self.array = array
        self.nodes = nodes.copy()
        self.update_id = update_id
        self.id = uuid.uuid4()
        self.precedent_id = precedent_id
        self.timestamp = datetime.datetime.now()
        self.setting = setting


class InitialEmbedding(Embedding):

    def __init__(self, array: np.float32, nodes: list(), update_id: uuid.UUID, precedent_id: uuid.UUID = None,
                 setting=None):

        super(InitialEmbedding, self).__init__(array, nodes, update_id, precedent_id, setting)


class DynamicEmbedding(Embedding):

    def __init__(self, array: np.float32, nodes: list(), update_id: uuid.UUID, precedent_id: uuid.UUID = None,
                 setting=None):

        super(DynamicEmbedding, self).__init__(array, nodes, update_id, precedent_id, setting)

