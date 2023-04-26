import rdflib


class Node:

    def __init__(self, value):

        self.value = value
        if self.value.startswith('_:'):
            self.node_type = 'bn'
        else:
            self.node_type = 'uri'

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