import json
from datetime import time

import torch
from abc import abstractmethod, ABC
from collections import OrderedDict
from logging import Logger
from typing import Iterable, List

from fastNLP import DataSet, Vocabulary
from fastNLP import DataSet as DataSetFastNlp
from fastNLP.embeddings import BertEmbedding
from fastNLP.modules.tokenizer.bert_tokenizer import WordpieceTokenizer
from tqdm import tqdm
from transformers import BertTokenizer

from TestLattice import load_cival_rules_rich_pretrain_word_list, rich_pretrain_word_path
from jsonl2json import get_bigrams
from lattice.utils_ import Trie, get_skip_path
from cival_wospert import util
from cival_wospert.entities_rules import Dataset, EntityType, RelationType, Entity, Relation, Document


# 输入读取模块
# 主要是对当前处理数据的类别 以及lattice词进行查询添加 形成数据token
class BaseInputReader(ABC):
    def __init__(self, types_path: str, tokenizer: WordpieceTokenizer, neg_entity_count: int = None,
                 neg_rel_count: int = None, max_span_size: int = None, logger: Logger = None):
        types = json.load(open(types_path,encoding='utf-8'), object_pairs_hook=OrderedDict)  # entity + relation types

        self._entity_types = OrderedDict()  # 构建有序字典用来存储实体和关系类别
        self._idx2entity_type = OrderedDict()
        self._relation_types = OrderedDict()
        self._idx2relation_type = OrderedDict()

        # entities
        # add 'None' entity type
        none_entity_type = EntityType('None', 0, 'None', 'No Entity')
        self._entity_types['None'] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        # specified entity types
        for i, (key, v) in enumerate(types['entities'].items()):
            entity_type = EntityType(key, i + 1, v['short'], v['verbose'])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i + 1] = entity_type

        # relations
        # add 'None' relation type
        none_relation_type = RelationType('None', 0, 'None', 'No Relation')
        self._relation_types['None'] = none_relation_type
        self._idx2relation_type[0] = none_relation_type

        # specified relation types
        for i, (key, v) in enumerate(types['relations'].items()):
            relation_type = RelationType(key, i + 1, v['short'], v['verbose'], v['symmetric'])
            self._relation_types[key] = relation_type
            self._idx2relation_type[i+1] = relation_type

        self._neg_entity_count = neg_entity_count
        self._neg_rel_count = neg_rel_count
        self._max_span_size = max_span_size

        self._datasets = dict()

        self._tokenizer = tokenizer
        self._logger = logger

        self._vocabulary_size = tokenizer.vocab_size
        self._context_size = -1

    @abstractmethod
    def read(self, datasets):
        pass

    def get_dataset(self, label) -> Dataset:
        return self._datasets[label]

    def get_entity_type(self, idx) -> EntityType:
        entity = self._idx2entity_type[idx]
        return entity

    def get_relation_type(self, idx) -> RelationType:
        try:
            relation = self._idx2relation_type[idx]
        except Exception as e:
            print(e)

        return relation

    def _calc_context_size(self, datasets: Iterable[Dataset]):
        sizes = []

        for dataset in datasets:
            for doc in dataset.documents:
                sizes.append(len(doc.encoding))

        context_size = max(sizes)
        return context_size

    def _log(self, text):
        if self._logger is not None:
            self._logger.info(text)

    @property
    def datasets(self):
        return self._datasets

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def relation_types(self):
        return self._relation_types

    @property
    def relation_type_count(self):
        return len(self._relation_types)

    @property
    def entity_type_count(self):
        return len(self._entity_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    @property
    def context_size(self):
        return self._context_size

    def __str__(self):
        string = ""
        for dataset in self._datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()

# 这是conll04数据的读取工具
class JsonInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None,
                 neg_rel_count: int = None, max_span_size: int = None, logger: Logger = None):
        super().__init__(types_path, tokenizer, neg_entity_count, neg_rel_count, max_span_size, logger)
        self.w_trie = Trie()
        self.get_w_tire()   # 通过ctb.50b词向量生成字典树
        self.bigram_vocab = Vocabulary()    # 声明一个词典 用来存储bigram
        self.raw_words = []
        self.words = []
        self.seq_len = []
        self.lexicons = []
        self.datasets_seq_fastnlp=DataSetFastNlp() # 构建char序列字符数据集
        self.jsonDataSetIndex = DataSetFastNlp()
        self.vocabulary = Vocabulary() # 构建字符和词的字典

    def get_w_tire(self):
        a = DataSet()
        w_list = load_cival_rules_rich_pretrain_word_list(rich_pretrain_word_path,
                                                          _refresh=False,
                                                          _cache_fp='cache/{}'.format("rules_lattice")
                                                          )
        for w in w_list:  # 构建词典树
            self.w_trie.insert(w)
        print(self.w_trie)
    def read(self, dataset_paths):
        for dataset_label, dataset_path in dataset_paths.items():
            dataset = Dataset(dataset_label, self._relation_types, self._entity_types, self._neg_entity_count,
                              self._neg_rel_count, self._max_span_size)
            self._parse_dataset(dataset_path, dataset)  # 真正的读取数据方法
            self._datasets[dataset_label] = dataset

        self._context_size = self._calc_context_size(self._datasets.values())



    def _parse_dataset(self, dataset_path, dataset):
        documents = json.load(open(dataset_path,encoding='utf-8'))
        count = 0
        raw_chars = []
        raw_chars_len = []
        entities = []
        relations = []
        # 初始化
        self.raw_words = []
        self.words = []
        self.seq_len = []
        self.lexicons = []
        for document in tqdm(documents, desc="构建词典中...Parse dataset '%s'" % dataset.label):

            # 构建词典和DataSetFastNlp用于后面的bert嵌入
            self._parse_document_fastNlpDataSet(document, dataset)
            # 按照document的数据样式处理document
            # self._parse_document(document, dataset)


        # 1.构建一个Dataset预备json
        jsonDataSet = {
                'raw_words':self.raw_words,
                'chars':self.words,
                'seq_len':self.seq_len,
                'lexicons':self.lexicons
        }


        self.datasets_seq_fastnlp = DataSetFastNlp(jsonDataSet)

        # 2.构建char字符数据集的字典
        # 这里进行公开数据集测试 先注释掉!!!!!!!!!
        # self.vocabulary =self.vocabulary.from_dataset(self.datasets_seq_fastnlp, field_name=['lexicons'])
        # print(self.vocabulary)
        # 3.吧搜索到的数据集中的所有词添加到bert的specialToken中
        wordList = []
        for index in self.vocabulary.idx2word:
            # print(self.vocabulary.idx2word[index])
            wordList.append(self.vocabulary.idx2word[index])
            if len(wordList) > 0: # 调整融入词典的大小 不同词典大小将直接决定训练速度
                break
        if dataset.label != 'test':
            self._tokenizer.add_special_tokens({'additional_special_tokens': wordList})
            print('特殊词典添加完成')
            print('当前词典大小为:'+ str(len(self._tokenizer)))


        # # 3.进行数据集的index转换
        # self.vocabulary.index_dataset(self.datasets_seq_fastnlp,
        #                                   field_name='chars', new_field_name='index_chars')
        # self.vocabulary.index_dataset(self.datasets_seq_fastnlp,
        #                                   field_name='lexicons', new_field_name='index_lexicons')
        # # 创建一个bertEmbedding工具用于准备将文本进行嵌入
        # bert_embedding = BertEmbedding(self.vocabulary, model_dir_or_name='E:\\NLP\\NER_RE\\spert-master\\scripts\\data\\models\\chinese_L-12_H-768_A-12', requires_grad=False,
        #                                word_dropout=0.01)
        # # 4.遍历数据 进行embedding
        # for item in tqdm(self.datasets_seq_fastnlp, desc="Parse dataset '%s'" % dataset.label):
        #     print(item)
        # # 再次处理序列 对其进行Bert序号的index嵌入
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):

            # 按照document的数据样式处理document
            self._parse_document(document, dataset)


    # 对文档中的数据进行单条的解析 每一条数据 获取其对应的lexicons
    def _parse_document_fastNlpDataSet(self, doc, dataset):
        jtokens = doc['tokens']
        jrelations = doc['relations']
        jentities = doc['entities']
        # 对序列进行拼接处理 并形成FastNlp DataSet标准格式
        jtokens_seq = ''.join(jtokens)
        # 获取序列长度
        seq_length = len(jtokens)
        self.raw_words.append(jtokens_seq)  # 存储到属性里 然后传出构成DataSet数据集
        self.words.append(jtokens)
        self.seq_len.append(seq_length)
        lexicons = get_skip_path(jtokens, self.w_trie)
        tempLexicons = list(map(lambda x: x[2], lexicons))
        self.lexicons.append(tempLexicons)



    def _parse_document(self, doc, dataset) -> Document:
        jtokens = doc['tokens']
        jrelations = doc['relations']
        jentities = doc['entities']

        # parse tokens
        doc_tokens, doc_encoding = self._parse_tokens_lattice(jtokens, dataset)
        # parse entity mentions
        entities = self._parse_entities(jentities, doc_tokens, dataset)

        # parse relations
        relations = self._parse_relations(jrelations, entities, dataset)

        # 1.获取bigram
        j_tokens_bigrams = get_bigrams(jtokens)
        doc_tokens_bigrams, doc_encoding_bigrams = None,None
        # doc_tokens_bigrams, doc_encoding_bigrams = self._parse_tokens(j_tokens_bigrams,dataset)

        # 2.根据字典树 构建lattice
        lexicons = get_skip_path(jtokens, self.w_trie)
        tempLexicons = list(map(lambda x: x[2], lexicons))
        self.lexicons.append(tempLexicons)
        lex_s = []
        lex_e = []
        for item in lexicons :
            lex_s.append(item[0])
            lex_e.append(item[1])
        lattice = jtokens + list(map(lambda x:x[2],lexicons))   # ['@', '14',  '消', '防', '',, '应', ', ''。', '14', '室外', '外给']
        # 2.1 转换lattice变为token
        lattice_tokens, lattice_encoding = self._parse_tokens_lattice(lattice, dataset)
        # 3.获取pos_s 和 pos_e
        temp_s = [0]
        temp_e = [0]
        pos_s = list(range(len(doc_encoding))) + lex_s
        pos_e = list(range(len(doc_encoding))) + lex_e
        list_zero_s = [0] * (len(lattice_encoding) - len(pos_s))
        list_zero_e = [0] * (len(lattice_encoding) - len(pos_e))
        pos_s = pos_s + list_zero_s
        pos_e = pos_e + list_zero_e
        # 验证数据
        # print("===" * 3)
        # print(doc_tokens)
        # print(entities[0].phrase)
        # print(entities[1].phrase)
        # print(doc_tokens[entities[0].span_start-1:entities[0].span_end-1])
        # print(doc_tokens[entities[1].span_start-1:entities[1].span_end-1])
        # create document
        # 判断lattice_encoding 和 pos_s 长度大小 进行噪声截断
        document = dataset.create_document(doc_tokens,
                                           doc_tokens_bigrams,
                                           entities,
                                           relations,
                                           doc_encoding,
                                           doc_encoding_bigrams,
                                           lattice_tokens,
                                           lattice_encoding,
                                           lex_s,
                                           lex_e,
                                           pos_s,
                                           pos_e)




        return document



    def _parse_tokens(self, jtokens, dataset):
        doc_tokens = []

        # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
        doc_encoding = [self._tokenizer.convert_tokens_to_ids('[CLS]')]
        # doc_encoding = []

        # parse tokens
        for i, token_phrase in enumerate(jtokens):
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))

            token = dataset.create_token(i, span_start, span_end, token_phrase)

            doc_tokens.append(token)
            doc_encoding += token_encoding

        # doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]

        return doc_tokens, doc_encoding

    # 针对lattice嵌入特殊设置的tokenize方式
    def _parse_tokens_lattice(self, jtokens, dataset):
        doc_tokens = []

        # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
        doc_encoding = [self._tokenizer.convert_tokens_to_ids('[CLS]')]
        # doc_encoding = []
        # self._tokenizer.add_special_tokens({'additional_special_tokens': ["室外", "@14"]})
        # parse tokens
        for i, token_phrase in enumerate(jtokens):
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            # 判断当没有对应token序号时 直接赋值0
            if len(token_encoding) == 0:
                token_encoding = [0]
            if len(token_encoding) == 1:
                token_encoding = token_encoding
            else:   #如果token 的长度不为1 就进行平均
                # Hash技巧  将词表中查询不到的特殊词token进行取余操作 使用bert原生词表中未使用的特殊token预留槽位
                temp_num = 1
                for item in token_encoding:
                    temp_num += item
                token_encoding = [temp_num / len(token_encoding)]


            # token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))

            token = dataset.create_token(i, span_start, span_end, token_phrase)

            doc_tokens.append(token)
            doc_encoding += token_encoding

        # doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]

        return doc_tokens, doc_encoding

    def _parse_entities(self, jentities, doc_tokens, dataset) -> List[Entity]:
        entities = []

        for entity_idx, jentity in enumerate(jentities):
            entity_type = self._entity_types[jentity['type']]
            start, end = jentity['start'], jentity['end']

            # create entity mention
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            entity = dataset.create_entity(entity_type, tokens, phrase)
            entities.append(entity)

        return entities

    def _parse_relations(self, jrelations, entities, dataset) -> List[Relation]:
        relations = []

        for jrelation in jrelations:
            relation_type = self._relation_types[jrelation['type']]

            head_idx = jrelation['head']
            tail_idx = jrelation['tail']

            # create relation
            try:
                head = entities[head_idx]
                tail = entities[tail_idx]
            except Exception as e:
                print('a')

            reverse = int(tail.tokens[0].index) < int(head.tokens[0].index)

            # for symmetric relations: head occurs before tail in sentence
            if relation_type.symmetric and reverse:
                head, tail = util.swap(head, tail)

            relation = dataset.create_relation(relation_type, head_entity=head, tail_entity=tail, reverse=reverse)
            relations.append(relation)

        return relations
