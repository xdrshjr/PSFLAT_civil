import collections
import copy
import math

import torch
from fastNLP.core.utils import _move_model_to_device, seq_len_to_mask
from torch import nn as nn
import torch.nn.functional as F
from torch.utils import data
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel

from cival_wospert import sampling
from cival_wospert import util
from cival_wospert.modules_lattice import Transformer_Encoder, get_embedding, Layer_Process, MultiHead_Attention_rel, \
    MultiHead_Attention_Lattice_rel_save_gpumm, Four_Pos_Fusion_Embedding, Positionwise_FeedForward


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token

    token_h = token_h[flat == token, :]



    return token_h


class cival_wospert(BertPreTrainedModel):
    """ Span-based model to jointly extract entities and relations """
    # spert的模型类
    VERSION = '1.1'

    def __init__(self, config: BertConfig, cls_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, max_pairs: int = 100):
        super(cival_wospert, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)
        self.max_seqlength = 512
        # layers
        self.rel_classifier = nn.Linear(config.hidden_size * 3 + size_embedding * 2, relation_types) # 关系分类器 3*768+25*2 → 5类关系 权重矩阵为weithg[2354,5] 假设batchsize为5,[2354,10]T * [2354,5] = [10,5]完成10个样本的5分类
        self.entity_classifier = nn.Linear(config.hidden_size * 2 + size_embedding, entity_types) # 实体分类器 2*768+25 → 5类实体 假设batchsize为5,[1561,10]T * [1561,5] = [10,5]完成10个样本的5分类
        self.size_embeddings = nn.Embedding(100, size_embedding) # 随机初始化一个[100,25]维的张量 用来当做额外的嵌入矩阵
        self.dropout = nn.Dropout(prop_drop)

        self._cls_token = cls_token
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs # 最大对数？ 这个比较疑惑

        # weight initialization
        self.init_weights() # 初始化模型参数
        # lattice机制参数
        self.seq_len = None
        self.lex_num = None
        self.hidden_size = config.hidden_size
        self.num_heads = 3
        self.num_layers = 1
        self.use_rel_pos = True
        self.learnable_position = True
        self.add_postition = False
        self.layer_preprocess_sequence = ''    # dropout
        self.layer_postprocess_sequence = 'an'  # norm & add 包括了残差连接和归一化
        self.rel_pos_init = 0
        self.pos_norm = True
        self.k_proj = True
        self.q_proj = True
        self.v_proj = True
        self.r_proj = True
        self.attn_ff = False
        self.ff_activate = 'relu'
        self.four_pos_fusion = 'attn'
        self.four_pos_fusion_shared = True
        self.four_pos_shared = False # 四向位置是否共享参数
        self.add_position = False
        if self.dropout is None:
            self.dropout = collections.defaultdict(int)
        self.scaled = True
        self.mode = collections.defaultdict(bool)
        self.dvc = torch.device('cuda')
        self.max_seq_len = 512
        self.ff_size = self.hidden_size
        # self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size * 4, self.hidden_size), nn.ReLU(inplace=True)).cuda()
        self.pos_fusion_forward = nn.Linear(self.hidden_size * 4, self.hidden_size)

        self.pos_fusion_forward = _move_model_to_device(self.pos_fusion_forward, device='cuda')
        # 此处不进行前处理
        self.layer_preprocess = Layer_Process(self.layer_preprocess_sequence, self.hidden_size, 0.5)
        # 进行输出后处理 add&norm
        self.layer_postprocess = Layer_Process(self.layer_postprocess_sequence, self.hidden_size, 0.3)

        if self.use_rel_pos:
            pe = get_embedding(self.max_seqlength,self.hidden_size,rel_pos_init=self.rel_pos_init)
            pe_sum = pe.sum(dim=-1,keepdim=True)
            if self.pos_norm:
                with torch.no_grad():
                    pe = pe/pe_sum
            self.pe = nn.Parameter(pe, requires_grad=self.learnable_position)
            if self.four_pos_shared:
                self.pe_ss = self.pe
                self.pe_se = self.pe
                self.pe_es = self.pe
                self.pe_ee = self.pe
            else:
                self.pe_ss = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
                self.pe_se = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
                self.pe_es = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
                self.pe_ee = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
        else:
            self.pe = None
            self.pe_ss = None
            self.pe_se = None
            self.pe_es = None
            self.pe_ee = None

        self.four_pos_fusion_embedding = \
            Four_Pos_Fusion_Embedding(self.pe,self.four_pos_fusion,self.pe_ss,self.pe_se,self.pe_es,self.pe_ee,
                                      self.max_seq_len,self.hidden_size,self.mode)
        self.attn = MultiHead_Attention_Lattice_rel_save_gpumm(self.hidden_size, self.num_heads,
                                                               pe=self.pe,
                                                               pe_ss=self.pe_ss,
                                                               pe_se=self.pe_se,
                                                               pe_es=self.pe_es,
                                                               pe_ee=self.pe_ee,
                                                               scaled=self.scaled,
                                                               mode=self.mode,
                                                               max_seq_len=self.max_seq_len,
                                                               dvc=self.dvc,
                                                               k_proj=self.k_proj,
                                                               q_proj=self.q_proj,
                                                               v_proj=self.v_proj,
                                                               r_proj=self.r_proj,
                                                               attn_dropout=0.2,  # dropout 比例
                                                               ff_final=self.attn_ff,
                                                               four_pos_fusion=self.four_pos_fusion)
        # 读取预先训练的后接模型
        # self.attn.load_state_dict(torch.load('E:\\NLP\\NER_RE\\spert-master\\scripts\\data\\save\\rules_datasets_train\\tempDir\\final_bert_model\\extra_att.state'))
        self.ff = Positionwise_FeedForward([self.hidden_size, self.hidden_size, self.hidden_size], self.dropout,
                                           ff_activate=self.ff_activate)
        # print('测试')





        if freeze_transformer:  # 是否需要冻结transformer的参数 这里不进行冻结
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def get_bert_block(self):
        print('保存TAPT模型：Bert')
        return self.bert

    def get_att_block(self):
        print('保存TAPT模型：Att')
        return self.attn

    def _forward_train(self,
                       task_name,
                       encodings: torch.tensor,
                       encodings_length: torch.tensor,
                       lattice_encodings: torch.tensor,
                       lattice_encodings_length: torch.tensor,
                       context_masks: torch.tensor,
                       context_masks_lattice: torch.tensor,
                       entity_masks: torch.tensor,
                       entity_masks_lattice: torch.tensor,
                       entity_sizes: torch.tensor,
                       relations: torch.tensor,
                       rel_masks: torch.tensor,
                       rel_masks_lattice: torch.tensor,
                       pos_ss_tensor: torch.tensor,
                       pos_se_tensor: torch.tensor,
                       pos_es_tensor: torch.tensor,
                       pos_ee_tensor:torch.tensor,
                       pos_e:torch.tensor,
                       pos_s:torch.tensor,
                       lex_num:torch.tensor
                       ):
        # get contextualized token embeddings from last transformer layer
        assert context_masks!=None
        assert context_masks_lattice!=None
        context_masks = context_masks.float()
        context_masks_lattice = context_masks_lattice.float()
        #pos_ss_tensor = pos_ss_tensor.float()
        self.max_seqlength = pos_s.size(1)
        batch_size = pos_s.size(0)
        # 得到原始序列长度和增加lattice之后的序列长度
        self.seq_len = encodings.shape[1]
        self.lex_num = lattice_encodings.shape[1] - encodings.shape[1]
        # 对位置信息使用相对位置嵌入的方式
        #这里的seq_len已经是之前的seq_len+lex_num了
        # self.pos_ss = pos_s.unsqueeze(-1)-pos_s.unsqueeze(-2)
        # self.pos_se = pos_s.unsqueeze(-1)-pos_e.unsqueeze(-2)
        # self.pos_es = pos_e.unsqueeze(-1)-pos_s.unsqueeze(-2)
        # self.pos_ee = pos_e.unsqueeze(-1)-pos_e.unsqueeze(-2)
        # self.pe_ss = get_embedding(self.max_seqlength,self.hidden_size,rel_pos_init=self.rel_pos_init)
        # self.pe_se = get_embedding(self.max_seqlength,self.hidden_size,rel_pos_init=self.rel_pos_init)
        # self.pe_es = get_embedding(self.max_seqlength,self.hidden_size,rel_pos_init=self.rel_pos_init)
        # self.pe_ee = get_embedding(self.max_seqlength,self.hidden_size,rel_pos_init=self.rel_pos_init)
        # pe_ss = self.pe_ss[(self.pos_ss).view(-1) + self.max_seqlength].view(size=[batch_size, self.max_seqlength, self.max_seqlength, -1])
        # pe_se = self.pe_se[(self.pos_se).view(-1) + self.max_seqlength].view(size=[batch_size, self.max_seqlength, self.max_seqlength, -1])
        # pe_es = self.pe_es[(self.pos_es).view(-1) + self.max_seqlength].view(size=[batch_size, self.max_seqlength, self.max_seqlength, -1])
        # pe_ee = self.pe_ee[(self.pos_ee).view(-1) + self.max_seqlength].view(size=[batch_size, self.max_seqlength, self.max_seqlength, -1])
        # #
        # pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1).cuda()
        # rel_pos_embedding = self.pos_fusion_forward(pe_4)


        # 这里可以确定是否使用lattice相对位置嵌入信息
        # 1.使用transformer进行位置的嵌入 拆分为4*111*111*768的矩阵进行嵌入
        # 2.或者直接使用bert输入原始序列和mask
        h = self.bert(input_ids=encodings, attention_mask=context_masks)[0]
        # h_lattice = self.bert(input_ids=lattice_encodings, attention_mask=context_masks_lattice)[0]

        # classify entities
        train_or_eval = 1   # 为1代表不经过后街Transfomer相对位置嵌入
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(train_or_eval,
                                                                encodings,
                                                                h,
                                                                entity_masks,
                                                                size_embeddings,
                                                                pos_s,
                                                                pos_e)
        # 是否采用lattice encoding的机制部分
        # entity_clf, entity_spans_pool = self._classify_entities(lattice_encodings, h_lattice, entity_masks_lattice, size_embeddings)
        # 测试代码1 取当前序列中所有span实体的类别
        tempEntity_clf = entity_clf.argmax(dim=-1)
        # print('==' * 10)
        # print('当前预测的所有span标签类别为:')
        # print(tempEntity_clf)
        # classify relations
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits

        tempEntity_clf_re = rel_clf.argmax(dim=-1)
        # print('==' * 10)
        # print('当前预测的所有关系类别预测为:')
        # print(tempEntity_clf_re)
        return entity_clf, rel_clf

    def _forward_train_only_entity_classify(self,
                       task_name,
                       encodings: torch.tensor,
                       encodings_length: torch.tensor,
                       lattice_encodings: torch.tensor,
                       lattice_encodings_length: torch.tensor,
                       context_masks: torch.tensor,
                       context_masks_lattice: torch.tensor,
                       entity_masks: torch.tensor,
                       entity_masks_lattice: torch.tensor,
                       entity_sizes: torch.tensor,
                       relations: torch.tensor,
                       rel_masks: torch.tensor,
                       rel_masks_lattice: torch.tensor,
                       pos_ss_tensor: torch.tensor,
                       pos_se_tensor: torch.tensor,
                       pos_es_tensor: torch.tensor,
                       pos_ee_tensor:torch.tensor,
                       pos_e:torch.tensor,
                       pos_s:torch.tensor,
                       lex_num:torch.tensor
                       ):
        # get contextualized token embeddings from last transformer layer
        assert context_masks!=None
        assert context_masks_lattice!=None
        context_masks = context_masks.float()
        context_masks_lattice = context_masks_lattice.float()
        #pos_ss_tensor = pos_ss_tensor.float()
        self.max_seqlength = pos_s.size(1)
        batch_size = pos_s.size(0)
        # 得到原始序列长度和增加lattice之后的序列长度
        self.seq_len = encodings.shape[1]
        self.lex_num = lattice_encodings.shape[1] - encodings.shape[1]
        # 对位置信息使用相对位置嵌入的方式
        #这里的seq_len已经是之前的seq_len+lex_num了
        # self.pos_ss = pos_s.unsqueeze(-1)-pos_s.unsqueeze(-2)
        # self.pos_se = pos_s.unsqueeze(-1)-pos_e.unsqueeze(-2)
        # self.pos_es = pos_e.unsqueeze(-1)-pos_s.unsqueeze(-2)
        # self.pos_ee = pos_e.unsqueeze(-1)-pos_e.unsqueeze(-2)
        # self.pe_ss = get_embedding(self.max_seqlength,self.hidden_size,rel_pos_init=self.rel_pos_init)
        # self.pe_se = get_embedding(self.max_seqlength,self.hidden_size,rel_pos_init=self.rel_pos_init)
        # self.pe_es = get_embedding(self.max_seqlength,self.hidden_size,rel_pos_init=self.rel_pos_init)
        # self.pe_ee = get_embedding(self.max_seqlength,self.hidden_size,rel_pos_init=self.rel_pos_init)
        # pe_ss = self.pe_ss[(self.pos_ss).view(-1) + self.max_seqlength].view(size=[batch_size, self.max_seqlength, self.max_seqlength, -1])
        # pe_se = self.pe_se[(self.pos_se).view(-1) + self.max_seqlength].view(size=[batch_size, self.max_seqlength, self.max_seqlength, -1])
        # pe_es = self.pe_es[(self.pos_es).view(-1) + self.max_seqlength].view(size=[batch_size, self.max_seqlength, self.max_seqlength, -1])
        # pe_ee = self.pe_ee[(self.pos_ee).view(-1) + self.max_seqlength].view(size=[batch_size, self.max_seqlength, self.max_seqlength, -1])
        #
        # pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1).cuda()
        # rel_pos_embedding = self.pos_fusion_forward(pe_4)


        # 这里可以确定是否使用lattice相对位置嵌入信息
        # 1.使用transformer进行位置的嵌入 拆分为4*111*111*768的矩阵进行嵌入
        # 2.或者直接使用bert输入原始序列和mask
        h = self.bert(input_ids=encodings, attention_mask=context_masks)[0]
        h_lattice = self.bert(input_ids=lattice_encodings, attention_mask=context_masks_lattice)[0]

        # encoded = self.encoder(h_lattice, lattice_encodings_length, lex_num=lex_num, pos_s=pos_s, pos_e=pos_e)

        # 进行模型输入
        # 1.1数据预处理 dropout
        # self.layer_preprocess(h_lattice)
        # output = self.attn(h_lattice, h_lattice, h_lattice, pe_4.size(1), pos_s=pos_s, pos_e=pos_e, lex_num=lex_num,
        #                    rel_pos_embedding=rel_pos_embedding)
        # batch_size = encodings.shape[0]

        # classify entities
        train_or_eval = 0   # 为1代表不经过后街Transfomer相对位置嵌入
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(train_or_eval,
                                                                lattice_encodings,
                                                                h_lattice,
                                                                entity_masks_lattice,
                                                                size_embeddings,
                                                                pos_s,
                                                                pos_e)

        return entity_clf

    def _forward_eval(self,
                      encodings: torch.tensor,
                      lattice_encodings: torch.tensor,
                      context_masks: torch.tensor,
                      context_masks_lattice: torch.tensor,
                      entity_masks: torch.tensor,
                      entity_masks_lattice: torch.tensor,
                      entity_sizes: torch.tensor,
                      entity_sizes_lattice: torch.tensor,
                      entity_spans: torch.tensor,
                      entity_sample_masks: torch.tensor,
                      pos_e: torch.tensor,
                      pos_s: torch.tensor
                      ):
        # get contextualized token embeddings from last transformer layer
        self.seq_len = encodings.shape[1]
        self.lex_num = lattice_encodings.shape[1] - encodings.shape[1]
        self.max_seqlength = pos_s.size(1)
        batch_size = pos_s.size(0)
        context_masks = context_masks.float()
        context_masks_lattice = context_masks_lattice.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)[0]
        # h = self.bert(input_ids=lattice_encodings, attention_mask=context_masks_lattice)[0] # h lattice
        batch_size = lattice_encodings.shape[0]
        ctx_size = context_masks.shape[-1]
        ctx_size_lattice = context_masks_lattice.shape[-1]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        train_or_eval = 1
        entity_clf, entity_spans_pool = self._classify_entities(train_or_eval,
                                                                encodings,
                                                                h,
                                                                entity_masks,
                                                                size_embeddings,
                                                                pos_s,
                                                                pos_e
                                                                )
        # 测试代码1 取当前序列中所有span实体的类别

        # print('==' * 10)
        # print('当前预测的所有span标签类别为:')

        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        relations, rel_masks, rel_sample_masks = self._filter_spans(entity_clf, entity_spans,
                                                                    entity_sample_masks, ctx_size)

        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            # apply sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        rel_clf = rel_clf * rel_sample_masks  # mask

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)
        tempEntity_clf = entity_clf.argmax(dim=-1)
        # print(tempEntity_clf)
        tempEntity_clf_rel = rel_clf.argmax(dim=-1)
        # print('==' * 10)
        # print('当前预测的所有span标签类别为:')
        # print(tempEntity_clf_rel)
        # print('==' * 10)

        return entity_clf, rel_clf, relations

    def _classify_entities(self, train_or_eval, encodings, h, entity_masks, size_embeddings, pos_s, pos_e):
        if train_or_eval == 0:
            # 1.相对位置嵌入获取四个位置矩阵

            rel_pos_embedding = self.four_pos_fusion_embedding(pos_s, pos_e)
            output = h
            output = self.layer_preprocess(output)
            self.lex_num_tensor = torch.tensor([self.lex_num, 0]).cuda()
            self.seq_len_tensor = torch.tensor([self.seq_len, 1]).cuda()
            output = self.attn(output, output, output, self.seq_len_tensor, pos_s=pos_s, pos_e=pos_e, lex_num=self.lex_num_tensor,
                               rel_pos_embedding=rel_pos_embedding)
            output = self.layer_postprocess(output)
            output = self.layer_preprocess(output)
            output = self.ff(output)
            output = self.layer_postprocess(output)
            # print("测试")











            # # 2.1 更换pe_4_tensor的维度
            # pe_4_tensor_trans = pe_4_tensor.permute(0, 1, 3, 2)
            # # 2.2 为h增加维度
            # h_unsqueezed = h.unsqueeze(2)
            # # 2.3 给h隐含向量进行相乘融入相对位置信息操作
            # h_equipRelativePos = torch.matmul(h_unsqueezed, pe_4_tensor_trans)
            #
            # h_equipRelativePos = F.softmax(h_equipRelativePos, dim = -1)
            # h_equipRelativePos = torch.matmul(h_equipRelativePos, pe_4_tensor).squeeze(2)
            # h_equipRelativePos = self.dropout(h_equipRelativePos)
            # # h_equipRelativePos = F.softmax(h_equipRelativePos, dim=-1)
            # # 关键语句 决定是否替换装备相对位置信息的h
            # # h_pre = h
            h = output


        # max pool entity candidate spans
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]

        # get cls token as candidate context representation
        entity_ctx = get_token(h, encodings, self._cls_token)

        # create candidate representations including context, max pooled span and size embedding
        entity_repr = torch.cat([entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                                 entity_spans_pool, size_embeddings], dim=2)
        entity_repr = self.dropout(entity_repr)

        # classify entity candidates
        entity_clf = self.entity_classifier(entity_repr)

        return entity_clf, entity_spans_pool

    def _classify_relations(self, entity_spans, size_embeddings, relations, rel_masks, h, chunk_start):
        batch_size = relations.shape[0]

        # create chunks if necessary
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :relations.shape[1], :]

        # get pairs of entity candidate representations
        entity_pairs = util.batch_index(entity_spans, relations)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        # get corresponding size embeddings
        size_pair_embeddings = util.batch_index(size_embeddings, relations)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # relation context (context between entity candidate pair)
        # mask non entity candidate tokens
        m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)
        rel_ctx = m + h

        # max pooling
        rel_ctx = rel_ctx.max(dim=2)[0]
        # set the context vector of neighboring or adjacent entity candidates to zero
        rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0

        # create relation candidate representations including context, max pooled entity candidate pairs
        # and corresponding size embeddings
        rel_repr = torch.cat([rel_ctx, entity_pairs, size_pair_embeddings], dim=2)
        rel_repr = self.dropout(rel_repr)

        # classify relation candidates
        chunk_rel_logits = self.rel_classifier(rel_repr)
        return chunk_rel_logits

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()  # get entity type (including none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # get spans classified as entities
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()
            # 筛选到这么多span是被分类为实体的[[24, 26], [13, 17], [30, 34], [35, 39], [7, 12]],其中[7,12]和[35,39]是真实的
            # create relations and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.rel_classifier.weight.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            if kwargs['task_name'] == 'entity':
                return self._forward_train_only_entity_classify(*args, **kwargs)
            elif kwargs['task_name'] == 'entity_relation':
                return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)


# Model access

_MODELS = {
    'cival_wospert': cival_wospert

}


def get_model(name):
    return _MODELS[name]
