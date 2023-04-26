import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from abc import ABC, abstractmethod
from pytorchtools import SparseDropout


class Navi(Module, ABC):

    def __init__(
        self, input_size, output_size, num_rel, bias=True, cuda=False
    ):
        super(Navi, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_rel = num_rel
        self.cuda = cuda
        self.instantiate_params(bias)
        self.reset_parameters()

    @abstractmethod
    def instantiate_params(self):
        pass

    @abstractmethod
    def reset_parameters(self):
        pass

    @abstractmethod
    def forward(self, features, adjacencies_T):
        return

class Navi_Diego(Navi):

    def __init__(self, input_size, output_size, num_rel, bias=True, cuda=False):

        super(Navi_Diego, self).__init__(input_size, output_size, num_rel, bias, cuda)

    def instantiate_params(self, bias):

        # R-GCN weights
        if self.cuda:
            self.w = Parameter(torch.cuda.FloatTensor(self.num_rel, self.input_size, self.output_size))
            # R-GCN bias
            if bias:
                self.bias = Parameter(torch.cuda.FloatTensor(self.num_rel, self.output_size))
            else:
                self.register_parameter("bias", None)

            self.w_t = Parameter(torch.cuda.FloatTensor(self.num_rel, self.input_size, self.output_size))
            # R-GCN bias
            if bias:
                self.bias_t = Parameter(torch.cuda.FloatTensor(self.num_rel, self.output_size))
            else:
                self.register_parameter("bias_t", None)

        else:

            self.w = Parameter(torch.FloatTensor(self.num_rel, self.input_size, self.output_size))
            # R-GCN bias
            if bias:
                self.bias = Parameter(torch.FloatTensor(self.num_rel, self.output_size))
            else:
                self.register_parameter("bias", None)

            self.w_t = Parameter(torch.FloatTensor(self.num_rel, self.input_size, self.output_size))
            # R-GCN bias
            if bias:
                self.bias_t = Parameter(torch.FloatTensor(self.num_rel, self.output_size))
            else:
                self.register_parameter("bias_t", None)

    def reset_parameters(self):

        for i in range(self.num_rel):
            nn.init.eye_(self.w.data[i])
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

        for i in range(self.num_rel):
            nn.init.eye_(self.w_t.data[i])
        if self.bias is not None:
            nn.init.zeros_(self.bias_t.data)

    def forward(self, features, adjacencies, adjacencies_t):


        degrees_lst = []
        out = torch.zeros(torch.Size((adjacencies.shape[1], features.shape[1])), dtype=torch.float32)

        for i in range(adjacencies.shape[0]):
            w_comp = torch.mm(torch.sparse.mm(adjacencies[i].float(), features), self.w[i])
            out += w_comp
            try:
                degrees = torch.sparse.sum(adjacencies[i], dim=1, dtype=torch.long).float().to_dense()
                b_comp = torch.mm(torch.diag(degrees), self.bias[i].repeat(adjacencies.shape[1], 1))
                degrees_lst.append(degrees)
                out += b_comp
            except RuntimeError as e:
                if e.args[0] != '_sparse_sum: sparse tensor input._nnz() == 0, please call torch.sparse.sum(input) instead.':
                    raise

            w_comp_t = torch.mm(torch.sparse.mm(adjacencies_t[i].float(), features), self.w_t[i])
            out += w_comp_t
            try:
                degrees = torch.sparse.sum(adjacencies_t[i], dim=1, dtype=torch.long).float().to_dense()
                b_comp_t = torch.mm(torch.diag(degrees), self.bias_t[i].repeat(adjacencies_t.shape[1], 1))
                degrees_lst.append(degrees)
                out += b_comp_t
            except RuntimeError as e:
                if e.args[0] != '_sparse_sum: sparse tensor input._nnz() == 0, please call torch.sparse.sum(input) instead.':
                    raise
        try:
            full_degrees = torch.sum(torch.stack(degrees_lst, dim=0), dim=0)
            full_degrees[full_degrees == 0] = 1
            full_degrees.pow_(-1)
        except RuntimeError:
            pass

        return torch.mm(torch.diag(full_degrees), out)


class Navi_Diego_alt(Navi):

    def __init__(self, input_size, output_size, num_rel, bias=True, cuda=False):

        super(Navi_Diego_alt, self).__init__(input_size, output_size, num_rel, bias, cuda)

    def instantiate_params(self, bias):

        # R-GCN weights
        if self.cuda:
            self.w = Parameter(torch.cuda.FloatTensor(self.num_rel, self.input_size, self.output_size))
            # R-GCN bias
            if bias:
                self.bias = Parameter(torch.cuda.FloatTensor(self.num_rel, self.output_size))
            else:
                self.register_parameter("bias", None)

            self.w_t = Parameter(torch.cuda.FloatTensor(self.num_rel, self.input_size, self.output_size))
            # R-GCN bias
            if bias:
                self.bias_t = Parameter(torch.cuda.FloatTensor(self.num_rel, self.output_size))
            else:
                self.register_parameter("bias_t", None)

        else:

            self.w = Parameter(torch.FloatTensor(self.num_rel, self.input_size, self.output_size))
            # R-GCN bias
            if bias:
                self.bias = Parameter(torch.FloatTensor(self.num_rel, self.output_size))
            else:
                self.register_parameter("bias", None)

            self.w_t = Parameter(torch.FloatTensor(self.num_rel, self.input_size, self.output_size))
            # R-GCN bias
            if bias:
                self.bias_t = Parameter(torch.FloatTensor(self.num_rel, self.output_size))
            else:
                self.register_parameter("bias_t", None)

    def reset_parameters(self):

        for i in range(self.num_rel):
            nn.init.eye_(self.w.data[i])
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

        for i in range(self.num_rel):
            nn.init.eye_(self.w_t.data[i])
        if self.bias is not None:
            nn.init.zeros_(self.bias_t.data)

    def forward(self, features, adjacencies, adjacencies_t):

        degrees_lst = []
        out = torch.zeros(torch.Size((adjacencies.shape[1], features.shape[1])), dtype=torch.float32)

        for i in range(adjacencies.shape[0]):

            w_comp = torch.sparse.mm(adjacencies[i].float(), features)

            try:

                degrees = torch.sparse.sum(adjacencies[i], dim=1, dtype=torch.long).float().to_dense()
                b_comp = torch.mm(torch.diag(degrees), self.bias[i].repeat(adjacencies.shape[1], 1))
                w_comp += b_comp

                degrees_tmp = torch.clone(degrees)
                degrees_tmp[degrees_tmp == 0] = 1
                degrees_tmp.pow_(-1)

                out += torch.mm(torch.mm(torch.diag(degrees_tmp), w_comp), self.w[i])

                degrees[degrees > 0] = 1
                degrees_lst.append(degrees)

            except RuntimeError:
                pass

            w_comp_t = torch.sparse.mm(adjacencies_t[i].float(), features)

            try:

                degrees = torch.sparse.sum(adjacencies_t[i], dim=1, dtype=torch.long).float().to_dense()
                b_comp_t = torch.mm(torch.diag(degrees), self.bias_t[i].repeat(adjacencies_t.shape[1], 1))
                w_comp_t += b_comp_t

                degrees_tmp = torch.clone(degrees)
                degrees_tmp[degrees_tmp == 0] = 1
                degrees_tmp.pow_(-1)

                out += torch.mm(torch.mm(torch.diag(degrees_tmp), w_comp_t), self.w_t[i])

                degrees[degrees > 0] = 1
                degrees_lst.append(degrees)

            except RuntimeError:
                pass

        full_degrees = torch.sum(torch.stack(degrees_lst, dim=0), dim=0)
        full_degrees[full_degrees == 0] = 1
        full_degrees.pow_(-1)

        return torch.mm(torch.diag(full_degrees), out)



# class Navi_Kroos(Navi):
#
#
#     def __init__(self, input_size, output_size, num_rel, bias=True, cuda=False):
#
#         super(Navi_Kroos, self).__init__(input_size, output_size, num_rel, bias, cuda)
#         self.degree_assigned = False
#         self.adjacencies_T = self.assign_degree()
#         self.degree_assigned = True
#
#     def assign_degree(self):
#
#         assert (not self.degree_assigned)
#
#         degree_supports = []
#         for j in range(self.num_rel):
#             degree_supports.append(torch.sum(self.adjacencies_T[j].to_dense(),0))
#
#         tmp = torch.zeros(degree_supports[0].shape, dtype=torch.float64)
#         for vec in degree_supports:
#             tmp += vec
#
#         adjacencies_T_new = []
#
#         for j in range(self.num_rel):
#             adjacencies_T_new.append((tmp[:,]*self.adjacencies_T[j].to_dense()).to_sparse())
#
#         return adjacencies_T_new
#
#     def instantiate_params(self, bias):
#
#         # R-GCN weights
#         self.w = Parameter(
#             torch.FloatTensor(self.num_rel, self.input_size, self.output_size)
#         )
#         # R-GCN bias
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(self.num_rel, self.output_size))
#         else:
#             self.register_parameter("bias", None)
#
#     def reset_parameters(self):
#
#         for i in range(self.num_rel):
#             nn.init.eye_(self.w.data[i])
#         if self.bias is not None:
#             nn.init.zeros_(self.bias.data)
#
#     def forward(self, features, adjacencies_T):
#
#         out_inp = torch.FloatTensor(torch.Size([adjacencies_T[0].shape[0], features.shape[1]]))
#         nn.init.zeros_(out_inp.data)
#         for i in range(self.num_rel):
#             out_inp += torch.mm(torch.sparse.mm(adjacencies_T[i], features), self.w[i])
#
#         out_bias = torch.FloatTensor(torch.Size([adjacencies_T[0].shape[0], features.shape[1]]))
#         nn.init.zeros_(out_bias.data)
#         occs_sum = torch.FloatTensor(torch.Size([adjacencies_T[0].shape[0]]))
#         nn.init.zeros_(occs_sum.data)
#         occs_sum = occs_sum.to_sparse()
#
#         for i in range(self.num_rel):
#             try:
#                 occurences = torch.sparse.sum(adjacencies_T[i], dim=1)
#             except RuntimeError:
#                 occurences = torch.sum(adjacencies_T[i].to_dense(), dim=1).to_sparse()
#             occs_sum += occurences
#             occ_matrix_sp = occurences.unsqueeze(1).to_dense().repeat(1, self.output_size).to_sparse()
#             diag = torch.diag(self.bias[i])
#             out_bias += torch.mm(occ_matrix_sp, diag.double())
#
#         occs_sum = occs_sum.to_dense()
#         occs_sum[occs_sum == 0] = 1
#         occs_sum_inv = 1. / occs_sum
#
#         out_comb = out_inp + out_bias
#
#         out = occs_sum_inv[:, None] * out_comb
#
#         return out
#
#
# class Navi_Kaiser(Navi):
#
#     def __init__(
#             self, input_size, output_size, X, adjacencies, num_rel, bias=True, cuda=False
#     ):
#         super(Navi_Kaiser, self).__init__(input_size, output_size, X, adjacencies, num_rel, bias, cuda)
#
#     def instantiate_params(self, bias):
#
#         # R-GCN weights
#         self.w = Parameter(
#             torch.FloatTensor(self.num_rel, self.input_size, self.output_size)
#         )
#
#         self.hadamards = Parameter(
#             torch.FloatTensor(self.num_rel,1)
#         )
#
#         # R-GCN bias
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(self.num_rel, self.output_size))
#         else:
#             self.register_parameter("bias", None)
#
#     def reset_parameters(self):
#
#         for i in range(self.num_rel):
#             nn.init.zeros_(self.w.data[i])
#         self.w.requires_grad = True
#
#         nn.init.ones_(self.hadamards)
#         self.hadamards.requires_grad = True
#
#         if self.bias is not None:
#             nn.init.zeros_(self.bias.data)
#         self.bias.requires_grad = True
#
#     def forward(self, features, adjacencies_T):
#
#         weights = self.w.view(self.w.shape[0] * self.w.shape[1], self.w.shape[2])
#
#         supports = []
#         occs = []
#         bias_terms = []
#
#         for i in range(self.num_rel):
#             a = torch.sparse.mm(adjacencies_T[i], features)
#             b = self.hadamards[i].repeat(adjacencies_T[i].shape[0])
#             supports.append(b[:, None]*a)
#
#         center_pre = torch.sum(torch.stack(supports), dim=0)
#
#         for i in range(self.num_rel):
#             try:
#                 occurences = torch.sparse.sum(adjacencies_T[i], dim=1).to_dense()
#             except RuntimeError:
#                 occurences = torch.sum(adjacencies_T[i].to_dense(), dim=1).to_sparse()
#             occs.append(occurences.repeat(adjacencies_T[i].shape[0]))
#             occ_matrix = occurences.unsqueeze(1).repeat(1, self.output_size)
#             diag = torch.diag(self.bias[i])
#             occ_matrix_sp = occ_matrix.to_sparse()
#             bias_terms.append(torch.mm(occ_matrix_sp, diag.double()))
#
#         occs_sum = torch.sum(torch.stack(occs), dim=0)
#         occs_sum[occs_sum == 0] = 1
#         occs_sum_inv = 1. / occs_sum
#
#         center = occs_sum_inv[:, None] * center_pre
#
#         supports_center = []
#
#         for i in range(self.num_rel):
#             supports_center.append(torch.sparse.mm(adjacencies_T[i], features) - center)
#
#         tmp = torch.cat(supports, dim=1)
#         out_inp = torch.mm(tmp, weights).double()
#
#         out_bias = torch.sum(torch.stack(bias_terms), dim=0)
#
#         out_comb = out_inp + out_bias
#
#         out = occs_sum_inv[:, None] * out_comb + center
#
#         return out
#
#
# class Navi_Iniesta(Navi):
#
#     def __init__(
#             self, input_size, output_size, features, adjacencies, num_rel, bias=True, cuda=False
#     ):
#         super(Navi_Iniesta, self).__init__(input_size, output_size, num_rel, bias, cuda)
#         self.relation_embeddings(features, adjacencies)
#
#     def relation_embeddings(self, features, adjacencies_T):
#
#         relation_supports = []
#         for j in range(self.num_rel):
#
#             indices = adjacencies_T[j]._indices().tolist()
#
#             indices_head = indices[1]
#             indices_tail = indices[0]
#
#             x = torch.cuda.FloatTensor(1, features.shape[1]) if self.cuda else torch.FloatTensor(1, features.shape[1])
#             nn.init.zeros_(x.data)
#             x.requires_grad = False
#
#             for i in range(len(indices_head)):
#                 tmp = features[indices_tail[i]]-features[indices_head[i]]
#                 x += tmp
#
#             relation_supports.append(x / len(indices_head))
#
#         self.relation_supports = relation_supports
#
#     def instantiate_params(self, bias):
#
#         # R-GCN weights
#         if self.cuda:
#             self.w = Parameter(
#                 torch.cuda.FloatTensor(self.num_rel, self.input_size, self.output_size)
#             )
#
#             # R-GCN bias
#             if bias:
#                 self.bias = Parameter(torch.cuda.FloatTensor(self.num_rel, self.output_size))
#             else:
#                 self.register_parameter("bias", None)
#         else:
#             self.w = Parameter(
#                 torch.FloatTensor(self.num_rel, self.input_size, self.output_size)
#             )
#
#             # R-GCN bias
#             if bias:
#                 self.bias = Parameter(torch.FloatTensor(self.num_rel, self.output_size))
#             else:
#                 self.register_parameter("bias", None)
#
#     def reset_parameters(self):
#
#         for i in range(self.num_rel):
#             nn.init.eye_(self.w.data[i])
#         self.w.requires_grad = True
#
#         if self.bias is not None:
#             nn.init.zeros_(self.bias.data)
#         self.bias.requires_grad = True
#
#     def forward(self, features, adjacencies_T):
#
#         out_inp = torch.cuda.FloatTensor(torch.Size([adjacencies_T[0].shape[0], features.shape[1]])) if self.cuda \
#             else torch.FloatTensor(torch.Size([adjacencies_T[0].shape[0], features.shape[1]]))
#         nn.init.zeros_(out_inp.data)
#
#         for i in range(self.num_rel):
#             Xpi = features + self.relation_supports[i]
#             out_inp += torch.mm(torch.sparse.mm(adjacencies_T[i], Xpi), self.w[i])
#
#         out_bias = torch.cuda.FloatTensor(torch.Size([adjacencies_T[0].shape[0], features.shape[1]])) if self.cuda \
#             else torch.FloatTensor(torch.Size([adjacencies_T[0].shape[0], features.shape[1]]))
#         nn.init.zeros_(out_bias.data)
#
#         occs_sum = torch.cuda.FloatTensor(torch.Size([adjacencies_T[0].shape[0]])) if self.cuda \
#             else torch.FloatTensor(torch.Size([adjacencies_T[0].shape[0]]))
#
#         nn.init.zeros_(occs_sum.data)
#         occs_sum = occs_sum.to_sparse()
#
#         for i in range(self.num_rel):
#             try:
#                 occurences = torch.sparse.sum(adjacencies_T[i], dim=1)
#             except RuntimeError:
#                 occurences = torch.sum(adjacencies_T[i].to_dense(), dim=1).to_sparse()
#             occs_sum += occurences
#             occ_matrix_sp = occurences.unsqueeze(1).to_dense().repeat(1, self.output_size).to_sparse()
#             diag = torch.diag(self.bias[i])
#             out_bias += torch.mm(occ_matrix_sp, diag.double())
#
#         out_comb = out_inp + out_bias
#
#         occs_sum = occs_sum.to_dense()
#         occs_sum[occs_sum == 0] = 1
#         occs_sum_inv = 1. / occs_sum
#
#         out = occs_sum_inv[:, None] * out_comb
#
#         return out
