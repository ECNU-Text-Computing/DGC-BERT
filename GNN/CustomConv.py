import torch
from dgl import DGLError
from dgl.nn.pytorch import APPNPConv, TAGConv, SAGEConv, GraphConv
from torch import nn
from dgl.nn.pytorch.conv.graphconv import EdgeWeightNorm
from torch.nn import functional as F
from dgl.utils import check_eq_shape, expand_as_pair


# def my_message_func(edges):
#     return {'m': edges.src['h'] * edges.data['w'].expand_as(edges.src['h'])}
#
#
# def my_reduce_func(nodes):
#     return {'h': torch.sum(nodes.mailbox['m'], dim=1)}

class CustomGraphConv(GraphConv):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(CustomGraphConv, self).__init__(in_feats, out_feats, norm, weight, bias, activation, allow_zero_in_degree)

    @staticmethod
    def my_message_func(edges):
        return {'m': edges.src['h'] * edges.data['_edge_weight'].expand_as(edges.src['h'])}

    @staticmethod
    def my_sum_func(nodes):
        return {'h': torch.sum(nodes.mailbox['m'], dim=1)}

    @staticmethod
    def my_copy_func(edges):
        return {'m': edges.src['h']}

    def forward(self, graph, feat, weight=None, edge_weight=None):
        r"""

        Description
        -----------
        Compute graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, which is the case for bipartite graph, the pair
            must contain two tensors of shape :math:`(N_{in}, D_{in_{src}})` and
            :math:`(N_{out}, D_{in_{dst}})`.
        weight : torch.Tensor, optional
            Optional external weight tensor.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature

        Raises
        ------
        DGLError
            Case 1:
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

            Case 2:
            External weight is provided while at the same time the module
            has defined its own weight parameter.

        Note
        ----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: :math:`(\text{in_feats}, \text{out_feats})`.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            aggregate_fn = self.my_copy_func
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = self.my_message_func

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = torch.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, self.my_sum_func)
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, self.my_sum_func)
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = torch.matmul(rst, weight)

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class CustomSAGEConv(SAGEConv):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(CustomSAGEConv, self).__init__(in_feats, out_feats, aggregator_type, feat_drop, bias, norm, activation)

    @staticmethod
    def my_message_func(edges):
        return {'m': edges.src['h'] * edges.data['_edge_weight'].expand_as(edges.src['h'])}

    @staticmethod
    def my_avg_func(nodes):
        return {'neigh': torch.mean(nodes.mailbox['m'], dim=1)}

    @staticmethod
    def my_sum_func(nodes):
        return {'neigh': torch.sum(nodes.mailbox['m'], dim=1)}

    @staticmethod
    def my_max_func(nodes):
        return {'neigh': torch.max(nodes.mailbox['m'], dim=1)[0]}

    @staticmethod
    def my_copy_func(edges):
        return {'m': edges.src['h']}

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N_{dst}, D_{out})`
            where :math:`N_{dst}` is the number of destination nodes in the input graph,
            :math:`D_{out}` is the size of the output feature.
        """
        self._compatibility_check()
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            msg_fn = self.my_copy_func
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                msg_fn = self.my_message_func

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == 'mean':
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                graph.update_all(msg_fn, self.my_avg_func)
                h_neigh = graph.dstdata['neigh']
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == 'gcn':
                check_eq_shape(feat)
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata['h'] = self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                else:
                    if graph.is_block:
                        graph.dstdata['h'] = graph.srcdata['h'][:graph.num_dst_nodes()]
                    else:
                        graph.dstdata['h'] = graph.srcdata['h']
                graph.update_all(msg_fn, self.my_sum_func)
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == 'pool':
                graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, self.my_max_func)
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
            elif self._aggre_type == 'lstm':
                graph.srcdata['h'] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == 'gcn':
                rst = h_neigh
            else:
                rst = self.fc_self(h_self) + h_neigh

            # bias term
            if self.bias is not None:
                rst = rst + self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


class CustomAPPNPConv(APPNPConv):
    def __init__(self,
                 k,
                 alpha,
                 edge_drop=0.):
        super(CustomAPPNPConv, self).__init__(k, alpha, edge_drop)

    # # @staticmethod
    # def my_message_func(self, edges):
    #     if (self._cur_k == 0) & (hasattr(self, 'rel_pos_embed')):
    #         pos_embed = self.rel_pos_embed[edges.dst['index'], edges.src['index']]
    #         return {'m': (edges.src['h'] + pos_embed) * edges.data['w'].expand_as(edges.src['h'])}
    #     else:
    #         return {'m': edges.src['h'] * edges.data['w'].expand_as(edges.src['h'])}
    @staticmethod
    def my_message_func(edges):
        return {'m': edges.src['h'] * edges.data['w'].expand_as(edges.src['h'])}

    @staticmethod
    def my_reduce_func(nodes):
        return {'h': torch.sum(nodes.mailbox['m'], dim=1)}

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute APPNP layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, *)`. :math:`N` is the
            number of nodes, and :math:`*` could be of any shape.
        edge_weight: torch.Tensor, optional
            edge_weight to use in the message passing process. This is equivalent to
            using weighted adjacency matrix in the equation above, and
            :math:`\tilde{D}^{-1/2}\tilde{A} \tilde{D}^{-1/2}`
            is based on :class:`dgl.nn.pytorch.conv.graphconv.EdgeWeightNorm`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *)` where :math:`*`
            should be the same as input shape.
        """
        with graph.local_scope():
            if edge_weight is None:
                src_norm = torch.pow(
                    graph.out_degrees().double().clamp(min=1), -0.5)
                shp = src_norm.shape + (1,) * (feat.dim() - 1)
                src_norm = torch.reshape(src_norm, shp).to(feat.device)
                dst_norm = torch.pow(
                    graph.in_degrees().double().clamp(min=1), -0.5)
                shp = dst_norm.shape + (1,) * (feat.dim() - 1)
                dst_norm = torch.reshape(dst_norm, shp).to(feat.device)
            else:
                edge_weight = EdgeWeightNorm(
                    'both')(graph, edge_weight)
            feat_0 = feat
            for _ in range(self._k):
                self._cur_k = _
                # normalization by src node
                if edge_weight is None:
                    feat = feat * src_norm
                graph.ndata['h'] = feat
                w = torch.ones(graph.number_of_edges(),
                            1) if edge_weight is None else edge_weight
                graph.edata['w'] = self.edge_drop(w).to(feat.device)
                graph.update_all(self.my_message_func,
                                 self.my_reduce_func)
                feat = graph.ndata.pop('h')
                # normalization by dst node
                if edge_weight is None:
                    feat = feat * dst_norm
                feat = (1 - self._alpha) * feat + self._alpha * feat_0
            return feat.float()


class CustomTAGConv(TAGConv):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=2,
                 bias=True,
                 activation=None,
                 ):
        super(CustomTAGConv, self).__init__(in_feats,
                                            out_feats,
                                            k,
                                            bias,
                                            activation)

    @staticmethod
    def my_copy_func(edges):
        return {'m': edges.src['h']}

    @staticmethod
    def my_message_func(edges):
        return {'m': edges.src['h'] * edges.data['_edge_weight'].expand_as(edges.src['h'])}

    @staticmethod
    def my_reduce_func(nodes):
        return {'h': torch.sum(nodes.mailbox['m'], dim=1)}

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute topology adaptive graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        edge_weight: torch.Tensor, optional
            edge_weight to use in the message passing process. This is equivalent to
            using weighted adjacency matrix in the equation above, and
            :math:`\tilde{D}^{-1/2}\tilde{A} \tilde{D}^{-1/2}`
            is based on :class:`dgl.nn.pytorch.conv.graphconv.EdgeWeightNorm`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        with graph.local_scope():
            assert graph.is_homogeneous, 'Graph is not homogeneous'
            if edge_weight is None:
                norm = torch.pow(graph.in_degrees().double().clamp(min=1), -0.5)
                shp = norm.shape + (1,) * (feat.dim() - 1)
                norm = torch.reshape(norm, shp).to(feat.device)

            msg_func = self.my_copy_func
            if edge_weight is not None:
                graph.edata["_edge_weight"] = EdgeWeightNorm(
                    'both')(graph, edge_weight)
                msg_func = self.my_message_func
            # D-1/2 A D -1/2 X
            fstack = [feat]
            for _ in range(self._k):
                if edge_weight is None:
                    rst = fstack[-1] * norm
                else:
                    rst = fstack[-1]
                graph.ndata['h'] = rst

                graph.update_all(msg_func,
                                 self.my_reduce_func)
                rst = graph.ndata['h']
                if edge_weight is None:
                    rst = rst * norm
                fstack.append(rst.float())

            rst = self.lin(torch.cat(fstack, dim=-1))

            if self._activation is not None:
                rst = self._activation(rst)

            return rst
