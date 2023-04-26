from fastapi import FastAPI, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import datetime
from navi_graph import NaviGraph
import torch
from node import Node
import json


class EmbeddingServer(FastAPI):
    """
    Class to deploy a SPARQL endpoint using an RDFLib Graph.
    """

    def __init__(self, navigraph, cors_enabled=True) -> None:
        """
        Constructor of the SPARQL endpoint, everything happens here.
        FastAPI calls are defined in this constructor
        """

        # Instantiate FastAPI
        super().__init__()
        self.navigraph = navigraph

        if cors_enabled:
            self.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        @self.post("/embed_sot")
        async def create_embedding(request: Request):
            """
            Send a SPARQL query to be executed through HTTP GET operation.
            \f
            :param request: The HTTP GET request
            :param query: SPARQL query input.
            """

            meta = await request.form()

            self.navigraph.embed_sot(ground_setting)

            return JSONResponse(
                status_code=200,
                content={"message": "Embedding was generated and stored.", "setting": str(meta)},
            )

        @self.post("/get_embedding")
        async def get_embedding(request: Request):
            """
            Send a SPARQL query to be executed through HTTP GET operation.
            \f
            :param request: The HTTP GET request
            :param query: SPARQL query input.
            """

            info = await request.json()

            # info = {'embed_type': 'trained', 'nodes': []}

            embed_type = info['embed_type']

            if embed_type in ['initial', 'trained']:
                if len(info['nodes']) == 0:
                    array = self.navigraph.reconstructor[embed_type].reconstruct_embeddings(self.navigraph)
                    array_np = array.cpu().detach().numpy().tolist()
                    return json.dumps({'nodes': [node.value for node in self.navigraph.nodes], 'array': array_np})
                else:
                    nodes = [Node(node) for node in info['nodes']]
                    try:
                        indices = [self.navigraph.nodes_dict[node] for node in nodes]
                    except KeyError:
                        return json.dumps(f'At least one node is not known to the graph')
                    array = self.navigraph.reconstructor[embed_type].reconstruct_embeddings(self.navigraph, indices=indices)
                    array_np = array.cpu().detach().numpy().tolist()
                    return json.dumps({'nodes': [node.value for node in nodes], 'array': array_np})

            elif embed_type == 'contextual':

                array = self.navigraph.embeddings[-1].array
                array_np = array.cpu().detach().numpy().tolist()

                if len(info['nodes']) == 0:
                    return json.dumps({'nodes': [node.value for node in self.navigraph.nodes], 'array': array_np})
                else:
                    nodes = [Node(node) for node in info['nodes']]
                    try:
                        indices = [self.navigraph.nodes_dict[node] for node in nodes]
                    except KeyError:
                        return json.dumps(f'At least one node is not known to the graph')
                    array_np_slice = [array_np[i] for i in indices]
                    return json.dumps({'nodes': [node.value for node in nodes], 'array': array_np_slice})

            return JSONResponse(
                status_code=400,
                content={"message": "An Error occured."},
            )

        @self.post("/update")
        async def insert_query(request: Request):

            print('\n################ START TRANSACTION ################')
            print(datetime.datetime.now().astimezone())

            results_request = await request.json()

            self.navigraph.update(results_request)

            print('\n~~~~~~~~~~~~~~~~~ END TRANSACTION ~~~~~~~~~~~~~~~~~')
            print(datetime.datetime.now().astimezone())

            return {"status": "SUCCESS"}

        @self.post("/put")
        async def insert_query(request: Request):

            print(datetime.datetime.now().astimezone())

            results = await request.json()

            # print(results)

            print('delete:')
            for result in results['delete']:
                print(result['s'], result['p'], result['o'])
            print('-----------------------------------------:')
            print('add:')
            for result in results['add']:
                print(result['s'], result['p'], result['o'])
            print('-----------------------------------------\n')

            return {
                "status": "SUCCESS",
                "data": results
            }


navigraph = NaviGraph()
navigraph.activate()
ground_setting = {'embedding engine': 'pykeen', 'model string': 'TransE', 'epochs': 1, 'dimension': 100}
# ground_setting = {'embedding engine': 'rdf2vec', 'epochs': 1, 'dimension': 100, 'depth': 4, 'number of walks': 20}
navigraph.embed_sot(ground_setting)
reconstructor_args = {'validate': True, 'use bias': True, 'epochs': 1, 'dropout': 0.4, 'lr': 0.001, 'cuda': True,
                      'loss_id': 'Inv_MSE', 'early_stop': 10, 'train_proportion': 0.9, 'manager_name': 'mourinho',
                      'navi layers': ['contextual'], 'dimension': 100,
                      'number relations': len(navigraph.relations_object)}
reconstructor_args['cuda'] = reconstructor_args['cuda'] and torch.cuda.is_available()
reconstructor_args['navi layers'] = sorted(reconstructor_args['navi layers'])

embedding_original = navigraph.embeddings[-1].array.cpu().detach().numpy()
navigraph.assign_reconstructor(reconstructor_args)
# embedding_reconstructed_0 = [regularize_navi(navigraph, initial=True, alpha=i).cpu().detach().numpy() for i in [0,0.5,1]]

navigraph.train_reconstructor()
# embedding_reconstructed_1 = [regularize_navi(navigraph, initial=False, alpha=i).cpu().detach().numpy() for i in [0,0.5,1]]

app = EmbeddingServer(
    navigraph=navigraph,
    cors_enabled=False
)
