import sys
import os

import dgl
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sys.path.append(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])

import torch
import transformers
import json
import pandas as pd
from main import get_DL_data, get_configs
from pretrained_models.scibert import SciBERT
from pretrained_models.dgc_bert import BAGIG
from pyecharts import options as opts
from pyecharts.charts import Graph

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams['font.size'] = 16


def get_gnn_atten(model, content, masks):
    model.eval()
    lengths = torch.sum(masks, dim=-1)
    content = content.permute(1, 0)

    output = model.bert(content, attention_mask=masks, return_dict=True, output_attentions=True,
                        output_hidden_states=True)
    pooled = output['pooler_output']

    # word_attention_gnn, semantic_attention_gnn = model.gnn(output, lengths)
    word_attention = torch.stack(output['attentions'][:3], dim=4).max(dim=4)[0]
    word_embed = model.gnn.hs_word_trans(torch.stack(output['hidden_states'][:3], dim=3).transpose(-2, -1)
                                         ).transpose(-2, -1).max(dim=3)[0]
    # semantic_attention = torch.stack(output['attentions'][-3:], dim=4).mean(dim=4)
    semantic_attention = torch.stack(output['attentions'][-3:], dim=4).max(dim=4)[0]
    semantic_embed = model.gnn.hs_semantic_trans(torch.stack(output['hidden_states'][-4:-1], dim=3).transpose(-2, -1)
                                                 ).transpose(-2, -1).max(dim=3)[0]
    word_result = gnn_forward(model.gnn.word_gnn, word_embed, word_attention, lengths)
    print(word_result[0])
    semantic_result = gnn_forward(model.gnn.semantic_gnn, semantic_embed, semantic_attention, lengths)
    return word_result, semantic_result


def gnn_forward(gnn, hidden_state, attention, lengths):
    batch_size = hidden_state.size(0)
    seq_len = hidden_state.size(1)
    dealt_atten = attention.mean(dim=1)
    # dealt_atten = attention.max(dim=1)[0]
    topk_result = torch.topk(dealt_atten, round(gnn.top_rate * hidden_state.size(1)))
    topk_values = topk_result.values
    topk_indices = topk_result.indices
    # # 新增加相对位置embed
    # if REL_POS:
    #     self.conv.rel_pos_embed = self.relative_position_embed(hidden_state.size(1), hidden_state.size(1))
    # sub_graphs = [gnn.seq_to_graph(topk_values[i], topk_indices[i], hidden_state[i], gnn.reduce, lengths[i]) for i in
    #               range(batch_size)]
    sub_graphs = [seq_to_graph_wz(gnn, topk_values[i], topk_indices[i], hidden_state[i], gnn.reduce, lengths[i]) for i
                  in
                  range(batch_size)]
    batch_graph = dgl.batch(sub_graphs)

    result_node_embedding = gnn.conv(batch_graph, batch_graph.ndata['h'])

    batch_graph.ndata['h'] = result_node_embedding
    unbatched_graph = dgl.unbatch(batch_graph)
    src = unbatched_graph[0].edges()[0].reshape(-1, 13)
    dst = unbatched_graph[0].edges()[1].reshape(-1, 13)
    weight = unbatched_graph[0].edata['w'].reshape(-1, 13)
    return src, dst, weight


def seq_to_graph_wz(gnn, topk_value, topk_indice, nodes, reduce='mean', length=256):
    '''
    输入的是单独的一个sample的边关系元组和隐藏层嵌入
    :param topk_value:
    :param topk_indice:
    :param reduce:
    :param nodes:
    :return:
    '''
    # edge_tuple = ([], [])
    # edge_weight = []
    # for ti in range(nodes.shape[0]):
    # for ti in range(length):
    #     # source_nodes = topk_indice[ti].cpu().numpy()
    #     # if reduce == 'softmax':
    #     #     # 防止top_k中取到0值
    #     #     pos_mask = torch.where(topk_value[ti] > 0, 0.0, -1e9)
    #     #     source_values = F.softmax(topk_value[ti] + pos_mask, dim=-1)
    #     # unmasked_index = torch.where(topk_indice[ti] < length)[0]
    #     # pos_index = torch.where(topk_value[ti] > 0)[0]
    #     pos_index = torch.nonzero(topk_value[ti], as_tuple=True)[0]
    #     source_nodes = topk_indice[ti][pos_index].cpu().numpy()
    #     if reduce == 'softmax':
    #         # 防止top_k中取到0值
    #         # pos_mask = torch.where(topk_value[ti][pos_index] > 0, 0.0, -1e9)
    #         # source_values = F.softmax(topk_value[ti][pos_index] + pos_mask, dim=-1)
    #         source_values = F.softmax(topk_value[ti][pos_index], dim=-1)
    #         edge_weight.append(source_values)
    #     edge_tuple[0].extend(source_nodes)
    #     edge_tuple[1].extend([ti] * len(source_nodes))
    valid_topk_value = topk_value[:length]
    valid_topk_indice = topk_indice[:length]
    top_num = valid_topk_indice.shape[1]
    target_nodes = torch.tensor([[i] * top_num for i in range(length)]).to(gnn.device)
    # 防止无用节点和padding节点信息的影响
    mask_edge_value = torch.softmax(torch.where(valid_topk_value > 0.0, valid_topk_value,
                                                torch.tensor(-1e9, dtype=torch.float).to(gnn.device)), dim=-1)
    # pos_sign = torch.nonzero(torch.flatten(mask_edge_value), as_tuple=True)[0]
    # source_nodes = torch.flatten(valid_topk_indice)[pos_sign]
    # target_nodes = torch.flatten(target_nodes)[pos_sign]
    source_nodes = torch.flatten(valid_topk_indice)
    target_nodes = torch.flatten(target_nodes)
    edge_tuple = (source_nodes, target_nodes)
    if reduce == 'softmax':
        # edge_weight = torch.flatten(mask_edge_value)[pos_sign]
        edge_weight = torch.flatten(mask_edge_value)

    sub_graph = dgl.graph(edge_tuple).to(gnn.device)
    # sub_graph.ndata['h'] = nodes
    sub_graph.ndata['h'] = gnn.node_trans(nodes[:length])
    # sub_graph.ndata['h'] = self.node_trans2(self.dropout(self.activation(self.node_trans1(nodes))))
    # sub_graph.ndata['index'] = torch.tensor(range(nodes.shape[0])).to(self.device)
    sub_graph.ndata['index'] = torch.tensor(range(length)).to(gnn.device)
    if reduce == 'mean':
        sub_graph.edata['w'] = torch.tensor([1] * len(edge_tuple[0]), dtype=torch.float32).to(gnn.device)
    elif reduce == 'softmax':
        # edge_weight = torch.cat(edge_weight)
        sub_graph.edata['w'] = edge_weight
    return sub_graph


def biaffine_attention(interaction, output, context, attention_mask):
    lengths = torch.sum(attention_mask, dim=-1)
    batch_size = output.shape[0]
    context_len = output.shape[1]
    # 减去padding对注意力机制造成的影响
    key_padding_mask = torch.tensor([[0] * length + [-1e9] * (context_len - length) for length in lengths]
                                    ).unsqueeze(dim=1).to(DEVICE)
    # attn = torch.matmul(self.output_trans(output), self.bilinear)
    attn = interaction.bilinear(interaction.output_trans(output))
    attn = torch.bmm(attn, interaction.context_trans(context).transpose(1, 2))
    attn = attn + interaction.U(output).expand(attn.shape) + interaction.V(context).transpose(1, 2).expand(attn.shape)
    attn = attn + key_padding_mask
    attn = torch.softmax(attn, dim=-1)
    mix = torch.bmm(attn, context)
    combined = torch.cat((output, mix), dim=2)
    # output = self.activation(self.fc(combined.view(-1, 2 * self.dim_model))).view(batch_size, -1, self.dim_model)
    output = torch.tanh(interaction.fc(combined.view(-1, 2 * interaction.dim_model))).view(batch_size, -1,
                                                                                           interaction.dim_model)
    # output = output.mean(dim=1)
    output = (torch.sum((output.transpose(1, 2) * attention_mask.unsqueeze(dim=1)).transpose(1, 2),
                        dim=1).T / lengths).T
    # output = output.max(dim=1)[0]

    return output, attn


def bagig_forward(model, content, masks):
    model.eval()
    lengths = torch.sum(masks, dim=-1)
    content = content.permute(1, 0)

    output = model.bert(content, attention_mask=masks, return_dict=True, output_attentions=True,
                        output_hidden_states=True)
    pooled = output['pooler_output']
    first_attn = output

    word_attention_gnn, semantic_attention_gnn = model.gnn(output, lengths)

    bert_out = model.bert_trans(pooled)
    # word_attention_gnn_mix = model.word_interaction(word_attention_gnn, semantic_attention_gnn, masks)
    word_attention_gnn_mix, word_attn = biaffine_attention(model.word_interaction, word_attention_gnn,
                                                           semantic_attention_gnn, masks)
    # semantic_attention_gnn_mix = model.semantic_interaction(semantic_attention_gnn, word_attention_gnn, masks)
    semantic_attention_gnn_mix, semantic_attn = biaffine_attention(model.semantic_interaction, semantic_attention_gnn,
                                                                   word_attention_gnn, masks)

    word_attention_gnn = model.word_gate(bert_out, word_attention_gnn_mix)
    semantic_attention_gnn = model.semantic_gate(bert_out, semantic_attention_gnn_mix)

    print('two-level', torch.cosine_similarity(word_attention_gnn, semantic_attention_gnn))
    print('word vs bert', torch.cosine_similarity(word_attention_gnn, bert_out))
    print('semantic vs bert', torch.cosine_similarity(bert_out, semantic_attention_gnn))
    gnn_out = torch.cat((word_attention_gnn, semantic_attention_gnn), dim=1)
    out = torch.cat((bert_out, gnn_out), dim=1)
    # print(out.shape)

    out = model.dropout(out)
    out = model.fc(out)
    out = torch.softmax(out, dim=-1)

    return out, word_attn, semantic_attn


def base_forward(model, content, lengths, masks, **kwargs):
    content = content.permute(1, 0)

    output = model.bert(content, attention_mask=masks)
    # print(len(output['hidden_states']))
    # print(output['hidden_states'][0].shape)
    # print(torch.cat(output['hidden_states'], dim=0).shape)
    out = output['pooler_output']
    word_weight = output['attentions'][0].mean(dim=1)
    semantic_wight = output['attentions'][-1].mean(dim=1)

    # out = output['last_hidden_state'][:, 1:, :]
    # out = (torch.sum(out, dim=1).T / (lengths - 1)).T
    # out = torch.mean(output['last_hidden_state'], dim=1)

    out = model.dropout(out)
    out = model.fc(out)
    # out = self.fc2(out)
    # out = F.softmax(out, dim=1)

    return out, word_weight, semantic_wight


def get_word_all_node_index(target, content, dataProcessor):
    word_list = dataProcessor.tokenizer.convert_ids_to_tokens(content, skip_special_tokens=False)
    word_index = [id for id, word in enumerate(word_list) if target in word]
    print(word_index)
    return word_index


def get_nodes_attn(index_list, word_result, semantic_result):
    result_dict = {}
    for index in index_list:
        result_dict[index] = {}
        result_dict[index]['word_source'] = word_result[0][index]
        result_dict[index]['word_weight'] = word_result[2][index]
        result_dict[index]['semantic_source'] = semantic_result[0][index]
        result_dict[index]['semantic_weight'] = semantic_result[2][index]
    return result_dict


def convert_node_to_word(node_list, content, dataProcessor):
    word_list = np.array(dataProcessor.tokenizer.convert_ids_to_tokens(content, skip_special_tokens=False))
    return word_list[node_list.detach().cpu().numpy()]


def get_node_print(node_list, node_dict, log=None, words=None):
    for index in node_list:
        if words:
            print('=' * 50, index, words[index], '=' * 50)
        else:
            print('=' * 50, index, '=' * 50)
        print(node_dict[index]['word_source'].detach().cpu().numpy())
        print(convert_node_to_word(node_dict[index]['word_source'], content, dataProcessor))
        print(node_dict[index]['semantic_source'].detach().cpu().numpy())
        print(convert_node_to_word(node_dict[index]['semantic_source'], content, dataProcessor))
        if log:
            if words:
                log.write('=' * 50 + str(index) + words[index] + '=' * 50 + '\n')
            else:
                log.write('=' * 50 + str(index) + '=' * 50 + '\n')
            log.write(str(node_dict[index]['word_source'].detach().cpu().numpy()) + '\n')
            log.write(str(convert_node_to_word(node_dict[index]['word_source'], content, dataProcessor)) + '\n')
            log.write(str(node_dict[index]['semantic_source'].detach().cpu().numpy()) + '\n')
            log.write(str(convert_node_to_word(node_dict[index]['semantic_source'], content, dataProcessor)) + '\n')


def get_result_heatmap(result, name):
    valid_nodes = len(result[0])
    adj_matrix = np.zeros((valid_nodes, valid_nodes))
    for index in range(valid_nodes):
        src_nodes = result[0][index]
        src_values = result[2][index]
        for i in range(len(src_nodes)):
            adj_matrix[src_nodes[i], index] = src_values[i]
    df = pd.DataFrame(index=list(range(valid_nodes)), columns=list(range(valid_nodes)), data=adj_matrix)
    plt.figure(dpi=300)
    sns.heatmap(data=df, cmap=plt.get_cmap("Blues"),
                xticklabels=30,
                yticklabels=30)
    plt.savefig('./img/{}.pdf'.format(name), bbox_inches='tight')
    return df


def get_heatmap(matrix, name):
    matrix = matrix.detach().cpu().numpy()
    df = pd.DataFrame(index=list(range(matrix.shape[0])), columns=list(range(matrix.shape[0])), data=matrix)
    plt.figure(dpi=300)
    sns.heatmap(data=df, cmap=plt.get_cmap("Blues"),
                xticklabels=30,
                yticklabels=30)
    plt.savefig('./img/{}.pdf'.format(name), bbox_inches='tight')
    return df


def get_k_hop_subgraph(result, target_index, k):
    src, dst, value = result
    node_dict = {}
    cur_nodes = [target_index]
    all_nodes = []
    for i in range(k):
        node_dict[i] = []
        for node in cur_nodes:
            node_dict[i].extend(src[node].detach().cpu().numpy())
        node_dict[i] = list(set(node_dict[i]))
        cur_nodes = node_dict[i]

    for i in range(k):
        all_nodes.extend(node_dict[i])
    all_nodes = list(set(all_nodes))
    all_nodes.sort()
    print(len(all_nodes))
    # node_dict['count'] = len(all_nodes)
    return src[all_nodes].detach().cpu().numpy(), dst[all_nodes].detach().cpu().numpy(), \
           value[all_nodes].detach().cpu().numpy(), node_dict, len(all_nodes)


def get_subgraph_img(result, file_name, target=0, words=None):
    srcs, dsts, value, nodes, count = result
    srcs = srcs.flatten()
    dsts = dsts.flatten()
    values = value.flatten()
    links_data = []
    for src, dst, value in zip(srcs, dsts, values):
        src_name = words[src] + '_' + str(src) if words else str(src)
        dst_name = words[dst] + '_' + str(dst) if words else str(dst)
        links_data.append(opts.GraphLink(source=src_name, target=dst_name, value=value))

    node_dict = {}
    for i in range(len(nodes)):
        hop = len(nodes) - (i + 1) + 1
        for node in nodes[hop - 1]:
            node_dict[node] = hop
    node_dict[target] = 0

    nodes_data = []
    for index in node_dict:
        name = words[index] + '_' + str(index) if words else str(index)
        nodes_data.append(opts.GraphNode(
            name=name,
            symbol_size=5 * (5 - node_dict[index]),
            category=node_dict[index],
            label_opts=opts.LabelOpts(
                # is_show=True if node_dict[index] < 2 else False,
                position='top',
                font_size=26 if node_dict[index] < 2 else 16 + 2*(4-node_dict[index]),
                font_weight='bold' if node_dict[index] < 2 else 'normal',
                font_family='Times new Romans'
            ),
        ))

    categories_data = []
    for category in range(len(nodes) + 1):
        if category == 0:
            categories_data.append(opts.GraphCategory(name='target node'.format(category)))
        else:
            categories_data.append(opts.GraphCategory(name='{}-hop'.format(category)))

    c = (
        Graph(opts.InitOpts(width='1200px', height="900px"))
            .add(
            "",
            nodes_data,
            links_data,
            categories_data,
            repulsion=300,
            layout="circular",
            linestyle_opts=opts.LineStyleOpts(color="source", curve=0.3, opacity=0.2),
            #             label_opts=opts.LabelOpts(is_show=False),
            is_rotate_label=True,

        )
            .set_global_opts(
            title_opts=opts.TitleOpts(title=file_name, pos_left='-30%'),
            legend_opts=opts.LegendOpts(orient="vertical", pos_left="-20%", pos_top="20%"),
            # legend_opts=opts.LegendOpts(orient="vertical", pos_left="2%", pos_top="20%",
            #                             textstyle_opts=opts.TextStyleOpts(
            #                                 font_size=26,
            #                                 font_weight='bold',
            #                                 font_family='Times new Romans'
            # )),
            toolbox_opts=opts.ToolboxOpts(
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                        background_color='rgb(0,0,0,0)',
                        pixel_ratio=4,
                    )
                )
            )
        )
            .render("./samples/{}.html".format(file_name))
    )


if __name__ == "__main__":
    data_source = 'PeerRead'
    seed = 333
    model_config = get_configs('PeerRead', ['BAGIG'])['BAGIG']
    pretrained_types = json.load(open('./configs/pretrained_types.json', 'r'))
    model_config['model_path'] = pretrained_types['SciBERT']['model_path']
    model_config['vocab_path'] = pretrained_types['SciBERT']['vocab_path']
    print(model_config)
    dataProcessor = get_DL_data(base_config=model_config, data_source=data_source,
                                BERT_tokenizer_path=model_config['vocab_path'],
                                load_data=model_config['saved_data'], seed=seed)
    # print(dataProcessor.get_dealt_text('you know this is test', 16, True)[0])
    model = torch.load('./checkpoints/PeerRead/BAGIG_top_biaffine+softmax_SciBERT_4_better_codes.pkl').cuda()
    # base = torch.load('./checkpoints/PeerRead/SciBERT_4.pkl').cuda()

    # print(type(model))
    # # print(model)
    # our_results = pd.read_csv('./checkpoints/results/PeerRead/mistake_results_BAGIG.csv')
    # base_results = pd.read_csv('./checkpoints/results/PeerRead/mistake_results_SciBERT.csv')
    # our_indexes = our_results['index'].to_list()
    # base_indexes = base_results['index'].to_list()
    # useful_indexes = [index for index in base_indexes if index not in our_indexes]
    # print(len(useful_indexes))
    # print(useful_indexes)
    # useful_results = base_results[base_results['index'].apply(lambda x: x in useful_indexes)]
    # print(useful_results)
    # useful_contents = useful_results['content'].to_list()
    #
    # cur_paper_index = 10115
    # cur_content = useful_results[useful_results['index'] == cur_paper_index]['content'].values[0]
    # print(cur_content)
    # content, length, masks = dataProcessor.get_dealt_text(cur_content.lower(), 256, True)
    # # print(dataProcessor.tokenizer.convert_ids_to_tokens(content, skip_special_tokens=False))
    # content = content.unsqueeze(dim=1).cuda()
    # masks = torch.tensor(masks).unsqueeze(dim=0).cuda()
    # print(content.shape)
    # print(masks.shape)
    # # _, base_word, base_semantic = base_forward(base, content, length, masks)
    # # get_heatmap(base_word.squeeze()[:length, :length].T, 'base_word_{}'.format(cur_paper_index))
    # # get_heatmap(base_semantic.squeeze()[:length, :length].T, 'base_semantic_{}'.format(cur_paper_index))
    #
    # # print(bagig_forward(model, content, masks)[0])
    # # _, word_attn, semantic_attn = bagig_forward(model, content, masks)
    # # word_attn = word_attn[0, :length, :length]
    # # semantic_attn = semantic_attn[0, :length, :length]
    # # df = get_heatmap(word_attn, 'word_biaffine_{}'.format(cur_paper_index))
    # # print(pd.DataFrame(df.sum(axis=0)).sort_values(by=0, ascending=False))
    # # df = get_heatmap(semantic_attn, 'semantic_biaffine_{}'.format(cur_paper_index))
    # # df.to_csv('sample_semantic.csv')
    # # print(pd.DataFrame(df.sum(axis=0)).sort_values(by=0, ascending=False))
    # words = dataProcessor.tokenizer.convert_ids_to_tokens(content, skip_special_tokens=False)
    #
    # print(our_results['probit'][0])
    # word_result, semantic_result = get_gnn_atten(model, content, masks)
    # # print(word_result[0][0])
    # # print(word_result[2][0])
    # # print(semantic_result[0][0])
    # # print(semantic_result[2][0])
    # df_word = get_result_heatmap(word_result, 'word_{}'.format(cur_paper_index))
    # df_semantic = get_result_heatmap(semantic_result, 'semantic_{}'.format(cur_paper_index))
    # important_words = ['task', 'approach', 'model', 'application', 'evaluation']
    # topic_words = ['semantic', 'ground', 'error']
    # node_list = [0]
    # for word in important_words:
    #     node_list.extend(get_word_all_node_index(word, content, dataProcessor))
    # attn_dict = get_nodes_attn(node_list, word_result, semantic_result)
    #
    # end_list = get_word_all_node_index('.', content, dataProcessor)
    # end_dict = get_nodes_attn(end_list, word_result, semantic_result)
    #
    # topic_list = []
    # for word in topic_words:
    #     topic_list.extend(get_word_all_node_index(word, content, dataProcessor))
    # # topic_list = get_word_all_node_index('online', content, dataProcessor) \
    # #              + get_word_all_node_index('algorithm', content, dataProcessor) \
    # #              + get_word_all_node_index('method', content, dataProcessor)
    # topic_dict = get_nodes_attn(topic_list, word_result, semantic_result)
    #
    # log = open('./img/{}.txt'.format(cur_paper_index), 'w+')
    # get_node_print(node_list, attn_dict, log, words)
    # print('*' * 100)
    # log.write('*' * 100 + '\n')
    # get_node_print(end_list, end_dict, log, words)
    # print('*' * 100)
    # log.write('*' * 100 + '\n')
    # get_node_print(topic_list, topic_dict, log, words)
    # log.close()
    # #
    # word_sub_result = list(get_k_hop_subgraph(word_result, 0, 2))
    # print(word_sub_result)
    # # json.dump(word_sub_result, open('./samples/word_{}.json'.format(cur_paper_index), 'w+'))
    # semantic_sub_result = list(get_k_hop_subgraph(semantic_result, 0, 3))
    #
    # get_subgraph_img(word_sub_result, 'word_{}'.format(cur_paper_index), words=words)
    # get_subgraph_img(semantic_sub_result, 'semantic_{}'.format(cur_paper_index), words=words)
    # for node in node_list:
    #     print(node)
    #     word_sub_result = list(get_k_hop_subgraph(word_result, node, 2))
    #     semantic_sub_result = list(get_k_hop_subgraph(semantic_result, node, 3))
    #     get_subgraph_img(word_sub_result, 'word_{}_{}_{}'.format(cur_paper_index, words[node], node), target=node, words=words)
    #     get_subgraph_img(semantic_sub_result, 'semantic_{}_{}_{}'.format(cur_paper_index, words[node], node), target=node, words=words)
    #
    # word_length = []
    # semantic_length = []
    # for i in range(10):
    #     word_length.append(list(get_k_hop_subgraph(word_result, 0, i))[-1])
    #     semantic_length.append(list(get_k_hop_subgraph(semantic_result, 0, i))[-1])
    # print(word_length)
    # print(semantic_length)

    # lengths = []
    # our_out = []
    # for content in useful_contents:
    #     # print(content)
    #     dealt_content, length, masks = dataProcessor.get_dealt_text(content.lower(), 256, True)
    #     dealt_content = dealt_content.unsqueeze(dim=1).cuda()
    #     masks = torch.tensor(masks).unsqueeze(dim=0).cuda()
    #     out = bagig_forward(model, dealt_content, masks)
    #     our_out.append(out.detach().cpu().numpy().tolist())
    #     print(length)
    #     lengths.append(length)

    # df = pd.DataFrame({'content': useful_contents, 'index': useful_results['index'],'length':lengths, 'our_probit': our_out, 'probit': useful_results['probit'],'true_label': useful_results['true_label']})
    # df.to_csv('lenghts_count.csv')
