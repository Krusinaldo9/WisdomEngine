import uuid
import datetime
from abc import ABC


class Update(ABC):

    # def __init__(self, deleted_triples, inserted_triples, id: uuid.UUID, predecessor_id: uuid.UUID = None):
    def __init__(self, id: uuid.UUID, predecessor_id: uuid.UUID = None):

        super(Update, self).__init__()

        self.start_time = datetime.datetime.now()
        self.end_time = None

        self.id = id
        self.predecessor_id = predecessor_id


class InitialUpdate(Update):

    def __init__(self, id: uuid.UUID, predecessor_id: uuid.UUID = None):

        super(InitialUpdate, self).__init__(id, predecessor_id)

        self.adjacencies = None
        self.data_values = None

    def finish_update(self, adjacencies, data_values):

        self.end_time = datetime.datetime.now()
        self.adjacencies = adjacencies
        self.data_values = data_values


class DynamicUpdate(Update):

    def __init__(self, id: uuid.UUID, predecessor_id: uuid.UUID = None):

        super(DynamicUpdate, self).__init__(id, predecessor_id)

        self.adjacencies_delete = None
        self.data_values_delete = None
        self.adjacencies_insert = None
        self.data_values_insert = None

    def finish_update(self, adjacencies_delete, data_values_delete, adjacencies_insert, data_values_insert):

        self.end_time = datetime.datetime.now()
        self.adjacencies_delete = adjacencies_delete
        self.data_values_delete = data_values_delete
        self.adjacencies_insert = adjacencies_insert
        self.data_values_insert = data_values_insert
