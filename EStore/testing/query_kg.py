from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper('http://localhost:2020/ds')
sparql.setQuery("""
    SELECT *
    WHERE {
        ?s ?p <http://example/Max> .
    }
    """
)
sparql.setReturnFormat(JSON)
try:
    ret = sparql.queryAndConvert()

    for r in ret["results"]["bindings"]:
        print(r)
except Exception as e:
    print(e)