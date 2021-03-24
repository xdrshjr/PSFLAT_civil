import copy
from collections import OrderedDict
from typing import List
from torch.utils.data import Dataset as TorchDataset
from fastNLP import DataSet as DataSetFastNLP

from cival_wospert import sampling

# 关系类的定义
class RelationType:
    def __init__(self, identifier, index, short_name, verbose_name, symmetric=False):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name   # 关系短名
        self._verbose_name = verbose_name   # 冗长名
        self._symmetric = symmetric # 是否对称

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

    @property
    def symmetric(self):
        return self._symmetric

    def __int__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, RelationType):
            return self._identifier == other._identifier
        return False

    def __hash__(self):
        return hash(self._identifier)


# 实体类的定义
class EntityType:
    def __init__(self, identifier, index, short_name, verbose_name):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name   # 实体短名
        self._verbose_name = verbose_name # 实体冗长名

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

    def __int__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, EntityType):
            return self._identifier == other._identifier
        return False

    def __hash__(self):
        return hash(self._identifier)


class Token:
    def __init__(self, tid: int, index: int, span_start: int, span_end: int, phrase: str):
        self._tid = tid  # ID within the corresponding dataset
        self._index = index  # original token index in document

        self._span_start = span_start  # start of token span in document (inclusive)
        self._span_end = span_end  # end of token span in document (exclusive)
        self._phrase = phrase

    @property
    def index(self):
        return self._index

    @property
    def span_start(self):
        return self._span_start

    @property
    def span_end(self):
        return self._span_end

    @property
    def span(self):
        return self._span_start, self._span_end

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Token):
            return self._tid == other._tid
        return False

    def __hash__(self):
        return hash(self._tid)

    def __str__(self):
        return self._phrase

    def __repr__(self):
        return self._phrase

class BigramToken(Token):
    def __init__(self):
        super().__init__()

class TokenSpan:
    def __init__(self, tokens):
        self._tokens = tokens

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    def __getitem__(self, s):
        if isinstance(s, slice):
            return TokenSpan(self._tokens[s.start:s.stop:s.step])
        else:
            return self._tokens[s]

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)


class Entity:
    def __init__(self, eid: int, entity_type: EntityType, tokens: List[Token], phrase: str):
        self._eid = eid  # ID within the corresponding dataset

        self._entity_type = entity_type

        self._tokens = tokens
        self._phrase = phrase

    def as_tuple(self):
        span_start_copy = copy.copy(self.span_start)
        span_end_copy = copy.copy(self.span_end)
        span_start_copy += 1
        span_end_copy += 1
        return span_start_copy, span_end_copy, self._entity_type    # 由于cls标签的引入  需要对实体span的起始和结束位置进行+1操作

    def as_tuple_evalute(self):
        span_start_copy = copy.copy(self.span_start)
        span_end_copy = copy.copy(self.span_end)
        return span_start_copy, span_end_copy, self._entity_type

    @property
    def entity_type(self):
        return self._entity_type

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self._eid == other._eid
        return False

    def __hash__(self):
        return hash(self._eid)

    def __str__(self):
        return self._phrase


class Relation:
    def __init__(self, rid: int, relation_type: RelationType, head_entity: Entity,
                 tail_entity: Entity, reverse: bool = False):
        self._rid = rid  # ID within the corresponding dataset
        self._relation_type = relation_type

        self._head_entity = head_entity
        self._tail_entity = tail_entity

        self._reverse = reverse

        self._first_entity = head_entity if not reverse else tail_entity
        self._second_entity = tail_entity if not reverse else head_entity

    def as_tuple(self):
        head = self._head_entity
        tail = self._tail_entity
        head_start, head_end = (head.span_start, head.span_end)
        tail_start, tail_end = (tail.span_start, tail.span_end)

        t = ((head_start, head_end, head.entity_type),
             (tail_start, tail_end, tail.entity_type), self._relation_type)
        return t

    @property
    def relation_type(self):
        return self._relation_type

    @property
    def head_entity(self):
        return self._head_entity

    @property
    def tail_entity(self):
        return self._tail_entity

    @property
    def first_entity(self):
        return self._first_entity

    @property
    def second_entity(self):
        return self._second_entity

    @property
    def reverse(self):
        return self._reverse

    def __eq__(self, other):
        if isinstance(other, Relation):
            return self._rid == other._rid
        return False

    def __hash__(self):
        return hash(self._rid)

# 单个数据条目的类
class Document:
    def __init__(self,
                 doc_id: int,
                 tokens: List[Token],
                 bigramtokens: List[Token],
                 entities: List[Entity],
                 relations: List[Relation],
                 encoding: List[int],
                 doc_encoding_bigrams:List[int],
                 lattice_tokens: List[Token],
                 lattice_encoding: List[int],
                 lex_s: List[int],
                 lex_e: List[int],
                 pos_s: List[int],
                 pos_e: List[int]
                 ):
        self._doc_id = doc_id  # ID within the corresponding dataset

        self._tokens = tokens
        self._entities = entities
        self._relations = relations

        # byte-pair document encoding including special tokens ([CLS] and [SEP])
        self._encoding = encoding
        # 根据lattice特殊构造的参数
        self._bigramtokens = bigramtokens
        self._doc_encoding_bigrams = doc_encoding_bigrams
        self._lattice_tokens = lattice_tokens
        self._lattice_encoding = lattice_encoding
        self._lex_s = lex_s
        self._lex_e = lex_e
        self._pos_s = pos_s
        self._pos_e = pos_e


    @property
    def doc_id(self):
        return self._doc_id

    @property
    def entities(self):
        return self._entities

    @property
    def relations(self):
        return self._relations

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def lattice_tokens(self):
        return TokenSpan(self._lattice_tokens)

    @property
    def encoding(self):
        return self._encoding

    @property
    def lattice_encoding(self):
        return self._lattice_encoding

    @property
    def lex_e(self):
        return self._lex_e

    @property
    def lex_s(self):
        return self._lex_s

    @property
    def pos_e(self):
        return self._pos_e

    @property
    def pos_s(self):
        return self._pos_s

    @encoding.setter
    def encoding(self, value):
        self._encoding = value



    def __eq__(self, other):
        if isinstance(other, Document):
            return self._doc_id == other._doc_id
        return False

    def __hash__(self):
        return hash(self._doc_id)


class BatchIterator:
    def __init__(self, entities, batch_size, order=None, truncate=False):
        self._entities = entities
        self._batch_size = batch_size
        self._truncate = truncate
        self._length = len(self._entities)
        self._order = order

        if order is None:
            self._order = list(range(len(self._entities)))

        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._truncate and self._i + self._batch_size > self._length:
            raise StopIteration
        elif not self._truncate and self._i >= self._length:
            raise StopIteration
        else:
            entities = [self._entities[n] for n in self._order[self._i:self._i + self._batch_size]]
            self._i += self._batch_size
            return entities


class Dataset(TorchDataset):
    TRAIN_MODE = 'train'
    EVAL_MODE = 'eval'

    def __init__(self, label, rel_types, entity_types, neg_entity_count,
                 neg_rel_count, max_span_size):
        self._label = label
        self._rel_types = rel_types
        self._entity_types = entity_types
        self._neg_entity_count = neg_entity_count
        self._neg_rel_count = neg_rel_count
        self._max_span_size = max_span_size
        self._mode = Dataset.TRAIN_MODE

        self._documents = OrderedDict()
        self._entities = OrderedDict()
        self._relations = OrderedDict()

        # current ids
        self._doc_id = 0
        self._rid = 0
        self._eid = 0
        self._tid = 0

        self.field_arrays = OrderedDict()

    def iterate_documents(self, batch_size, order=None, truncate=False):
        return BatchIterator(self.documents, batch_size, order=order, truncate=truncate)

    def iterate_relations(self, batch_size, order=None, truncate=False):
        return BatchIterator(self.relations, batch_size, order=order, truncate=truncate)

    def create_token(self, idx, span_start, span_end, phrase) -> Token:
        token = Token(self._tid, idx, span_start, span_end, phrase)
        self._tid += 1
        return token

    def create_document(self,
                        tokens,
                        bigramtokens,
                        entity_mentions,
                        relations,
                        doc_encoding,
                        doc_encoding_bigrams,
                        lattice_tokens,
                        lattice_encoding,
                        lex_s,
                        lex_e,
                        pos_s,
                        pos_e) -> Document:
        document = Document(self._doc_id,
                            tokens,
                            bigramtokens,
                            entity_mentions,
                            relations,
                            doc_encoding,
                            doc_encoding_bigrams,
                            lattice_tokens,
                            lattice_encoding,
                            lex_s,
                            lex_e,
                            pos_s,
                            pos_e
                            )
        self._documents[self._doc_id] = document
        self._doc_id += 1
        self.field_arrays = self.documents
        return document

    def create_entity(self, entity_type, tokens, phrase) -> Entity:
        mention = Entity(self._eid, entity_type, tokens, phrase)
        self._entities[self._eid] = mention
        self._eid += 1
        return mention

    def create_relation(self, relation_type, head_entity, tail_entity, reverse=False) -> Relation:
        relation = Relation(self._rid, relation_type, head_entity, tail_entity, reverse)
        self._relations[self._rid] = relation
        self._rid += 1
        return relation

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, index: int):
        doc = self._documents[index]
        # 抽样测试
        # # 1.打印整个句子的token
        # temp = doc.tokens._tokens
        # print(temp)
        # # 2.打印取得的实体
        # entity01 = doc.entities[0].phrase
        # entity02 = doc.entities[1].phrase
        # entity01_span_s = doc.entities[0].span_start
        # entity01_span_e = doc.entities[0].span_end
        # entity02_span_s = doc.entities[1].span_start
        # entity02_span_e = doc.entities[1].span_end
        # print("实体1:" + entity01)
        # print("实体2:" + entity02)
        # print("索引取得的实体1:" + str(temp[entity01_span_s - 1:entity01_span_e - 1]))
        # print("索引取得的实体1:" + str(temp[entity02_span_s - 1:entity02_span_e - 1]))
        if self._mode == Dataset.TRAIN_MODE:
            return sampling.create_train_sample(doc, self._neg_entity_count, self._neg_rel_count,
                                                self._max_span_size, len(self._rel_types))
        else:
            return sampling.create_eval_sample(doc, self._max_span_size)

    def switch_mode(self, mode):
        self._mode = mode

    @property
    def label(self):
        return self._label

    @property
    def input_reader(self):
        return self._input_reader

    @property
    def documents(self):
        return list(self._documents.values())

    @property
    def entities(self):
        return list(self._entities.values())

    @property
    def relations(self):
        return list(self._relations.values())

    @property
    def document_count(self):
        return len(self._documents)

    @property
    def entity_count(self):
        return len(self._entities)

    @property
    def relation_count(self):
        return len(self._relations)
