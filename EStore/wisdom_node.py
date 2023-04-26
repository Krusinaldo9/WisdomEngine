import rdflib
import numpy as np
import uuid

class NodeInformation:

    def __init__(self, structure_degree: int, data_degree: int, vector: np.float32, update_log_id: uuid.UUID = None):

        self.structure_degree = structure_degree
        self.data_degree = data_degree
        self.degree = structure_degree + data_degree

        if self.degree > 0:
            self.active = True
        else:
            self.active = False

        self.vector = vector
        self.update_log_id = update_log_id

class Wisdom_Node:

    def __init__(self, value):

        self.value = value
        if self.value.startswith('_:'):
            self.node_type = 'bn'
        else:
            self.node_type = 'uri'

        self.node_informations = list()

    def ni(self):

        if len(self.node_informations) > 0:
            return self.node_informations[-1]
        else:
            return None

    def update_ni(self, node_information: NodeInformation):

        self.node_informations.append(node_information)

    def __eq__(self, other):
        return (self.node_type, self.value) == (other.node_type, other.value)

    def __hash__(self):
        return hash((self.node_type, self.value))

    def __str__(self):
        return f'Node {self.node_type}: {self.value}'

    def __repr__(self):
        return f'Node {self.node_type}: {self.value}'

    def node2rdflib(self):
        return rdflib.term.URIRef(self.value)

    def update_information(self, node_information):
        self.node_informations.append(node_information)