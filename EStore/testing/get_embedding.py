import requests
import json
import numpy as np

arr_full = []

data = {
    'embed_type': 'trained',
    'nodes': ['http://www.aifb.uni-karlsruhe.de/Publikationen/viewPublikationOWL/id122instance', 'http://example/Max', 'http://example/Moritz']
}
# data = {
#     'embed_type': 'trained',
#     'nodes': []
# }

r = requests.post('http://localhost:5000/get_embedding', json=data)
res = json.loads(r.json())
try:
    arr = np.asarray(res['array'])
    arr_full.append(arr)
except TypeError:
    print(res)

data['embed_type'] = 'initial'
r = requests.post('http://localhost:5000/get_embedding', json=data)
res = json.loads(r.json())
try:
    arr = np.asarray(res['array'])
    arr_full.append(arr)
except TypeError:
    print(res)


data['embed_type'] = 'contextual'
r = requests.post('http://localhost:5000/get_embedding', json=data)
res = json.loads(r.json())
try:
    arr = np.asarray(res['array'])
    arr_full.append(arr)
except TypeError:
    print(res)

for arr in arr_full:
    print(arr)
