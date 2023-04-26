# Teaming AI: Wisdom Engine Prototype

## 1. Triple Store (TStore)

The triple store is based on a Fuseki triple store from the Apache Jena framework (https://jena.apache.org/). Further, the rdf-delta (https://afs.github.io/rdf-delta/) extension was integrated and adapted allow for synchronized with the new component, i.e., with the embedding store.

### Installation

The triple store was designed and tested in Java for the target bytecode version 11. The external libraries used in the program can be imported by means of the corresponding pom.xml file. After doing so, the triple store can be started by running the Server.java file. Initially, it will contain the benchmark aifb knowledge graph.

## 2. Embedding Store (EStore)

The embedding store is a python-based FastAPI server that contains methods for generating state of the art embeddings, as well as Navi reconstruction formalisms. 

### Installation

The code was designed in Python 3.7 and the libraries used therein can be found in requirements.txt. *After* a TStore was initialized, the embedding store can be run via server_run.py. After the creation of an initial embedding and a Navi reconstruction, the embedding store is synchronized with the triplestore via the rdf-delta extension so that updates in the graph will be forwarded to the embedding store. Accordingly, the embeddings will be updated.

## Testing/Demo

The interaction with an agent can be simulated via the testing files in the EStore directory. Thus, the running TStore can be updated (and thus also the EStore), the KG can be queried, and the initial/updated embeddings can be retrieved. As an example, we considered the insertion of to new people Max & Moritz and linked them to some existing publications as authors. However, these examples can be arbitrarily adjusted.
