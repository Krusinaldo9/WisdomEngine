import csv
import sys
import os

def create_tsv(graph, tmp_file_full, tmp_file_test):

    triples = []

    for s, p, o in graph:
        triples.append([s.toPython(), p.toPython(), o.toPython()])

    with open(tmp_file_full, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(triples)

    with open(tmp_file_test, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(triples[:2])


if __name__ == "__main__":

    id = sys.argv[1]
    create_tsv(id, '../../Data/')
