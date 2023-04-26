import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import time
from pytorchtools import EarlyStopping
from abc import ABC, abstractmethod
import navis
from losses import assign_loss_function
import sys, os
from pytorchtools import SparseDropout
import random
import math


class NaviManager(Module, ABC):

    def __init__(self, args, initial):

        super(NaviManager, self).__init__()

        self.args = args
        self.cuda = args['cuda']

        self.layers = nn.ModuleList()

        self.sparse_dropper = SparseDropout()
        self.best_val = -1
        self.initial = initial
        if initial:
            self.dropout = 0

        self.num_rel = args['number relations']
        self.input_size = self.output_size = args['dimension']
        self.cuda = args['cuda']
        self.use_bias = args['use bias']
        self.activate()

    def add_navi(self, navitype):

        if navitype == 'contextual':
            self.layers.append(navis.Navi_Diego(self.input_size, self.output_size,
                                                self.num_rel, bias=self.use_bias, cuda=self.cuda))
        elif navitype == 'translational':
            self.layers.append(navis.Navi_Iniesta(self.input_size, self.output_size,
                                                  self.num_rel, bias=self.use_bias, cuda=self.cuda))
        elif navitype == 'contextual_alt':
            self.layers.append(navis.Navi_Diego_alt(self.input_size, self.output_size,
                                              self.num_rel, bias=self.use_bias, cuda=self.cuda))
        else:
            print(f'The specified Navi {navitype} is not known.')
            assert False

    def activate(self):

        for navi in self.args['navi layers']:
            self.add_navi(navi)
        self.instantiate_params(self.use_bias)
        self.reset_parameters()

        if not self.initial:
            self.set_training_method(self.args)


    def set_training_method(self, args):

        assert not self.initial

        self.lr = args['lr']
        self.early_stop = args['early_stop']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.args['validate']:
            self.early_stopping = EarlyStopping(patience=self.early_stop, verbose=False)
        self.validate = args['validate']
        self.use_bias = args['use bias']
        self.epochs = args['epochs']
        self.dropout = args['dropout']
        self.loss_id = args['loss_id']
        self.criterion = assign_loss_function(self.loss_id)
        self.train_proportion = args['train_proportion']

    def num_navis(self):
        return len(self.layers)

    @abstractmethod
    def instantiate_params(self, bias):
        pass

    @abstractmethod
    def reset_parameters(self):
        pass

    @abstractmethod
    def forward(self, X, adjacencies=None):
        return

    def do_train(self, array, adjacencies, batch_size=32):

        adjacencies_t = torch.sparse_coo_tensor(torch.index_select(adjacencies._indices(), 0, torch.tensor([0, 2, 1])),
                                                adjacencies._values(), adjacencies.shape)

        assert not self.initial

        X = array.cuda() if self.cuda else array

        idx = list(range(X.size(dim=0)))

        if self.validate:
            random.shuffle(idx)
            length_train = int(len(idx) * self.train_proportion)
            self.split_idx = {'train': idx[:length_train], 'test': idx[length_train:]}
        else:
            self.split_idx = {'train': idx, 'test': []}

        epoch = 1
        train_bool = True
        early_stop = False

        with torch.no_grad():
            self.eval()

            emb_valid = self(X, adjacencies, adjacencies_t)
            loss_val = self.criterion(
                emb_valid[self.split_idx['test']], X[self.split_idx['test']]
            )

            if loss_val <= self.best_val or self.best_val == -1:
                self.best_val = loss_val
                self.model_state = {
                    "state_dict": self.state_dict(),
                    "best_val": loss_val,
                    "best_epoch": epoch,
                    "optimizer": self.optimizer.state_dict(),
                }
            print("Initial Loss of the network on the {num} test data: loss: {loss}".format(
                num=len(self.split_idx['test']),
                loss=loss_val.item()))

        while train_bool and epoch <= self.epochs:

            # if (epoch) % 50 == 0:
            #     self.save_checkpoint()

            # Start training
            self.train()

            idx_tmp = self.split_idx['train'].copy()
            random.shuffle(idx_tmp)
            num_batches = math.ceil(len(idx_tmp) / batch_size)

            for j, i in enumerate(range(0, len(idx_tmp), batch_size)):

                sys.stdout.write("\r" + f'Epoch {epoch}. Batch {j+1} of {num_batches} \t')
                sys.stdout.flush()

                batch_idx = idx_tmp[i:i + batch_size]

                a1 = self.get_adjacencies_drop(adjacencies)
                a2 = self.get_adjacencies_drop(adjacencies_t)

                emb_train = self(X, self.get_adjacencies_drop(adjacencies), self.get_adjacencies_drop(adjacencies_t),
                                 batch_idx)
                loss = self.criterion(emb_train[list(range(len(batch_idx)))], X[batch_idx])
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            t = time.time()

            # print("Training Loss on {num} training data: {loss}".format(
            #         epoch=epoch, num=len(self.split_idx['train']), loss=str(loss.item())),
            #     time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)))

            if self.validate:
                with torch.no_grad():
                    self.eval()
                    emb_valid = self(X, adjacencies, adjacencies_t)
                    loss_val = self.criterion(
                        emb_valid[self.split_idx['test']], X[self.split_idx['test']]
                    )

                    print("Loss of the network on the {num} test data: loss: {loss}".format(
                        num=len(self.split_idx['test']),
                        loss=loss_val.item()))

                    if loss_val <= self.best_val or self.best_val == -1:
                        self.best_val = loss_val
                        self.model_state = {
                            "state_dict": self.state_dict(),
                            "best_val": loss_val,
                            "best_epoch": epoch,
                            "optimizer": self.optimizer.state_dict(),
                        }

                    self.early_stopping(loss_val, self)
                    if self.early_stopping.early_stop:
                        print("Early stopping")
                        train_bool = False
                        early_stop = True
                        continue

            train_bool = True
            epoch += 1

        self.log_dict = {'early_stop': early_stop, 'best_loss': self.model_state['best_val'].item(),
                    'epoch': self.model_state['best_epoch']}

    def create_new_embeddings(self, skg_1, row_switch_matrix, reduced_adjacencies):

        with torch.no_grad():
            self.eval()
            old_embedding = torch.from_numpy(skg_1.embedding).cuda() if self.cuda \
                else torch.from_numpy(skg_1.embedding)
            row_switch_csr = csr2tensor(row_switch_matrix, self.cuda)
            X = torch.mm(row_switch_csr, old_embedding.double())

            del old_embedding, row_switch_csr

            adjacencies = self.get_adjacencies_drop(reduced_adjacencies, skg_1.object_relations)

            embeddings = self(adjacencies=adjacencies, X=X)

            return embeddings.cpu().detach().numpy()

    def reconstruct_embeddings(self, navigraph, adjacencies=None, adjacencies_t=None, indices=None, embedding_tmp=None):

        with torch.no_grad():
            self.eval()
            if embedding_tmp is None:
                X = navigraph.embeddings[-1].array.cuda() if self.cuda else navigraph.embeddings[-1].array
            else:
                X = embedding_tmp.cuda() if self.cuda else embedding_tmp

            if adjacencies is None or adjacencies_t is None:

                adjacencies_t = torch.sparse_coo_tensor(
                    torch.index_select(navigraph.adjacencies._indices(), 0, torch.tensor([0, 2, 1])),
                    navigraph.adjacencies._values(), navigraph.adjacencies.shape)

                embeddings = self(X=X, adjacencies=navigraph.adjacencies, adjacencies_t=adjacencies_t, indices=indices)

            else:

                embeddings = self(X=X, adjacencies=adjacencies, adjacencies_t=adjacencies_t, indices=indices)

            return embeddings

    def get_adjacencies_drop(self, adjacencies):

        adjacencies_drop = self.sparse_dropper(adjacencies, self.dropout, self.training)
        return adjacencies_drop


class Manager_Mourinho(NaviManager):

    def __init__(self, args, initial):

        super(Manager_Mourinho, self).__init__(args, initial)

    def instantiate_params(self, bias):
        pass

    def reset_parameters(self):
        pass

    def forward(self, X, adjacencies, adjacencies_t, indices=None):

        if indices is not None:

            number_indices = len(indices)
            selector_indices = torch.IntTensor(np.array([[i, x] for i, x in enumerate(indices)])).T
            selector_values = torch.ByteTensor(np.array([1] * number_indices))
            selector_shape = torch.Size((number_indices, adjacencies.shape[1]))
            selector = torch.sparse_coo_tensor(selector_indices, selector_values, selector_shape).float()

            adjacencies = torch.stack([torch.sparse.mm(selector, adjacencies[i].float()) for i in range(adjacencies.shape[0])],
                                      dim=0).byte()
            adjacencies_t = torch.stack([torch.sparse.mm(selector, adjacencies_t[i].float()) for i in range(adjacencies_t.shape[0])],
                                      dim=0).byte()

        supports = []
        for k in range(self.num_navis()):
            x = self.layers[k](X, adjacencies, adjacencies_t)
            supports.append(x)

        navi_results = [res.cuda() if res is not None and self.cuda else res for res in supports]

        out = torch.mean(torch.stack(navi_results), dim=0)

        return out


class Manager_Ferguson(NaviManager):

    def __init__(self, args, initial):

        super(Manager_Ferguson, self).__init__(args, initial)

    def instantiate_params(self, bias):

        # R-GCN weights
        if self.cuda:
            self.w = Parameter(torch.cuda.DoubleTensor(self.num_navis(), self.input_size, self.output_size))

            # R-GCN bias
            if bias:
                self.bias = Parameter(torch.cuda.DoubleTensor(self.output_size))
            else:
                self.register_parameter("bias", None)
        else:
            self.w = Parameter(torch.DoubleTensor(self.num_navis(), self.input_size, self.output_size))

            # R-GCN bias
            if bias:
                self.bias = Parameter(torch.DoubleTensor(self.output_size))
            else:
                self.register_parameter("bias", None)

    def reset_parameters(self):

        for i in range(self.num_navis()):
            nn.init.eye_(self.w.data[i])
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, X, adjacencies):

        supports = []
        for k in range(self.num_navis()):
            x = self.layers[k](X, adjacencies)
            supports.append(x)

        navi_results = [res.cuda() if res is not None and self.cuda else res for res in supports]

        weights = self.w.view(
            self.w.shape[0] * self.w.shape[1], self.w.shape[2]
        )

        tmp = torch.cat(navi_results, dim=1)
        out = torch.mm(tmp, weights) + self.bias.unsqueeze(0)
        out = out / (len(navi_results))

        return out


class Manager_Wenger(NaviManager):

    def __init__(self, args, initial):

        super(Manager_Wenger, self).__init__(args, initial)

    def instantiate_params(self, bias):

        # R-GCN weights
        self.hadamard_vectors = Parameter(
            torch.DoubleTensor(self.num_navis(), self.output_size)
        )

        # R-GCN bias
        if bias:
            print("This Manager can't be used with a bias.")
            self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):

        for i in range(self.num_navis()):
            nn.init.ones_(self.hadamard_vectors.data[i])

    def forward(self, X, adjacencies):

        adjacencies = self.get_adjacencies_drop(adjacencies)

        adjacencies_t = torch.sparse_coo_tensor(torch.index_select(adjacencies._indices(), 0, torch.tensor([0, 2, 1])),
                                                adjacencies._values(), adjacencies.shape)

        supports = []
        for k in range(self.num_navis()):
            x = self.layers[k](X, adjacencies, adjacencies_t)
            supports.append(x)

        navi_results = [res.cuda() if res is not None and self.cuda else res for res in supports]

        supports_hadamard = []
        for k in range(self.num_navis()):
            supports_hadamard.append(navi_results[k] * self.hadamard_vectors[k][:, None].T)

        out = torch.mean(torch.stack(supports_hadamard), dim=0)

        return out


class Manager_Heynckes(NaviManager):

    def __init__(self, args, initial):

        super(Manager_Heynckes, self).__init__(args, initial)

    def instantiate_params(self, bias):

        # R-GCN weights
        self.w = Parameter(
            torch.DoubleTensor(self.num_navis(), self.input_size, self.output_size)
        )

        # R-GCN bias
        if bias:
            self.bias = Parameter(torch.DoubleTensor(self.output_size))
        else:
            self.register_parameter("bias", None)

        self.w_rgcn = Parameter(
            torch.DoubleTensor(self.num_rel*2, self.output_size, self.output_size)
        )
        # R-GCN bias
        if bias:
            self.bias_rgcn = Parameter(torch.DoubleTensor(self.num_rel*2, self.output_size))
        else:
            self.register_parameter("bias_rgcn", None)

    def reset_parameters(self):

        for i in range(self.num_navis()):
            nn.init.eye_(self.w.data[i])
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

        for i in range(self.num_rel*2):
            nn.init.zeros_(self.w_rgcn.data[i])
        if self.bias is not None:
            nn.init.zeros_(self.bias_rgcn.data)

    def forward(self, X, adjacencies):

        adjacencies = self.get_adjacencies_drop(adjacencies)

        supports = []
        for k in range(self.num_navis()):
            x = self.layers[k](X, adjacencies)
            supports.append(x)

        navi_results = [res.cuda() if res is not None and self.cuda else res for res in supports]

        weights = self.w.view(
            self.w.shape[0] * self.w.shape[1], self.w.shape[2]
        )

        tmp = torch.cat(navi_results, dim=1)
        out = torch.mm(tmp, weights).double() + self.bias.unsqueeze(0)
        out_own = out / (len(navi_results))

        out_inp = torch.DoubleTensor(torch.Size([adjacencies[0].shape[0], X.shape[1]]))
        nn.init.zeros_(out_inp.data)

        for i in range(self.num_rel*2):
            out_inp += torch.mm(torch.sparse.mm(adjacencies[i], out_own), self.w_rgcn[i])

        out_bias = torch.DoubleTensor(torch.Size([adjacencies[0].shape[0], X.shape[1]]))
        nn.init.zeros_(out_bias.data)
        occs_sum = torch.DoubleTensor(torch.Size([adjacencies[0].shape[0]]))
        nn.init.zeros_(occs_sum.data)
        occs_sum = occs_sum.to_sparse()

        for i in range(self.num_rel*2):
            try:
                occurences = torch.sparse.sum(adjacencies[i], dim=1)
            except RuntimeError:
                occurences = torch.sum(adjacencies[i].to_dense(), dim=1).to_sparse()
            occs_sum += occurences
            occ_matrix_sp = occurences.unsqueeze(1).to_dense().repeat(1, self.output_size).to_sparse()
            diag = torch.diag(self.bias_rgcn[i])
            out_bias += torch.mm(occ_matrix_sp, diag.double())

        occs_sum = occs_sum.to_dense()
        occs_sum[occs_sum == 0] = 1
        occs_sum_inv = 1. / occs_sum

        out_comb = out_inp + out_bias

        out = torch.index_select(out_own, 0, torch.tensor([self.nodes_dict[t] for t in self.targets])) + occs_sum_inv[:, None] * out_comb

        return out


def to_sparse(x):
    """converts dense tensor x to sparse format"""
    x_typename = torch.typename(x).split(".")[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def csr2tensor(A, cuda):
    coo = A.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    shape = coo.shape

    if cuda:
        out = torch.sparse.DoubleTensor(i, v, torch.Size(shape)).cuda()
    else:
        out = torch.sparse.DoubleTensor(i, v, torch.Size(shape))
    return out


def get_manager(args, initial):

    manager_name = args['manager_name']

    if manager_name == 'mourinho':
        manager = Manager_Mourinho(args, initial)
    elif manager_name == 'ferguson':
        manager = Manager_Ferguson(args, initial)
    elif manager_name == 'wenger':
        manager = Manager_Wenger(args, initial)
    elif manager_name == 'heynckes':
        manager = Manager_Heynckes(args, initial)
    else:
        sys.exit('Manager', manager_name, 'is not known. Leaving the program...')

    return manager
