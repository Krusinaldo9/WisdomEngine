from SPARQLWrapper import SPARQLWrapper

sparql = SPARQLWrapper('http://localhost:2020/ds')

sparql.setQuery("""
    PREFIX : <http://example/>
    INSERT DATA {
        <http://www.aifb.uni-karlsruhe.de/Publikationen/viewPublikationOWL/id122instance> <http://swrc.ontoware.org/ontology#author> :Max.
        <http://www.aifb.uni-karlsruhe.de/Publikationen/viewPublikationOWL/id504instance> <http://swrc.ontoware.org/ontology#author> :Moritz.
    }
    """)

results = sparql.query()
print(results.response.read())
