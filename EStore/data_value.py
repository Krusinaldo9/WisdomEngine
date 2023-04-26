import rdflib

class DataValue:

    def __init__(self, data_type, value):

        self.data_type = data_type
        self.value = value

    def __eq__(self, other):

        return (self.data_type, self.value) == (other.data_type, other.value)

    def __hash__(self):
        return hash((self.data_type, self.value))

    def __str__(self):
        return f'Data Value {self.data_type}: {self.value}'

    def __repr__(self):
        return f'Data Value {self.data_type}: {self.value}'

    def literal2rdflib(self):
        return rdflib.term.Literal(self.value, datatype=self.data_type)