import random

import torch
from torch import nn

from cival_wospert import util


def create_train_sample(doc, neg_entity_count: int, neg_rel_count: int, max_span_size: int, rel_type_count: int):
    # assert len(doc.encoding) < 512
    encodings = doc.encoding
    # 1.1特殊构建的lattice机制嵌入
    lattice_encodings = doc.lattice_encoding

    # 获取encodings和lattice_encodings的长度
    encodings_length = len(doc.encoding)
    lattice_encodings_length = len(lattice_encodings)


    token_count = len(doc.tokens)
    context_size = len(encodings)
    context_size_lattice = len(lattice_encodings)

    # positive entities
    pos_entity_spans, pos_entity_types, pos_entity_masks, pos_entity_sizes = [], [], [], []
    pos_entity_masks_lattice = []
    pos_entity_sizes_lattice = []

    try:
        for e in doc.entities:
            pos_entity_spans.append(e.span)
            pos_entity_types.append(e.entity_type.index)
            pos_entity_masks.append(create_entity_mask(*e.span, context_size))  # span的context上下文mask
            pos_entity_masks_lattice.append(create_entity_mask(*e.span, context_size_lattice))
            pos_entity_sizes.append(len(e.tokens))
            pos_entity_sizes_lattice.append(len(e.tokens))
    except Exception as e:
        print(e)
    # positive relations
    pos_rels, pos_rel_spans, pos_rel_types, pos_rel_masks, pos_rel_masks_lattice = [], [], [], [], []
    for rel in doc.relations:
        s1, s2 = rel.head_entity.span, rel.tail_entity.span
        pos_rels.append((pos_entity_spans.index(s1), pos_entity_spans.index(s2)))
        pos_rel_spans.append((s1, s2))
        pos_rel_types.append(rel.relation_type)
        pos_rel_masks.append(create_rel_mask(s1, s2, context_size))
        pos_rel_masks_lattice.append(create_rel_mask(s1, s2, context_size_lattice))

    # negative entities
    neg_entity_spans, neg_entity_sizes = [], []
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            if span not in pos_entity_spans:
                neg_entity_spans.append(span)
                neg_entity_sizes.append(size)

    # sample negative entities
    neg_entity_samples = random.sample(list(zip(neg_entity_spans, neg_entity_sizes)),
                                       min(len(neg_entity_spans), neg_entity_count))
    neg_entity_spans, neg_entity_sizes = zip(*neg_entity_samples) if neg_entity_samples else ([], [])

    neg_entity_masks = [create_entity_mask(*span, context_size) for span in neg_entity_spans]
    neg_entity_masks_lattice = [create_entity_mask(*span, context_size_lattice) for span in neg_entity_spans]
    neg_entity_types = [0] * len(neg_entity_spans)

    # negative relations   负采样关系
    # use only strong negative relations, i.e. pairs of actual (labeled) entities that are not related
    neg_rel_spans = []

    for i1, s1 in enumerate(pos_entity_spans):
        for i2, s2 in enumerate(pos_entity_spans):
            rev = (s2, s1)
            rev_symmetric = rev in pos_rel_spans and pos_rel_types[pos_rel_spans.index(rev)].symmetric

            # do not add as negative relation sample:
            # neg. relations from an entity to itself
            # entity pairs that are related according to gt
            # entity pairs whose reverse exists as a symmetric relation in gt
            if s1 != s2 and (s1, s2) not in pos_rel_spans and not rev_symmetric:
                neg_rel_spans.append((s1, s2))

    # sample negative relations
    neg_rel_spans = random.sample(neg_rel_spans, min(len(neg_rel_spans), neg_rel_count))

    neg_rels = [(pos_entity_spans.index(s1), pos_entity_spans.index(s2)) for s1, s2 in neg_rel_spans]
    neg_rel_masks = [create_rel_mask(*spans, context_size) for spans in neg_rel_spans]
    neg_rel_masks_lattice = [create_rel_mask(*spans, context_size_lattice) for spans in neg_rel_spans]
    neg_rel_types = [0] * len(neg_rel_spans)

    # merge
    entity_types = pos_entity_types + neg_entity_types
    entity_masks = pos_entity_masks + neg_entity_masks
    entity_masks_lattice = pos_entity_masks_lattice + neg_entity_masks_lattice
    entity_sizes = pos_entity_sizes + list(neg_entity_sizes)

    rels = pos_rels + neg_rels
    rel_types = [r.index for r in pos_rel_types] + neg_rel_types
    rel_masks = pos_rel_masks + neg_rel_masks
    rel_masks_lattice = pos_rel_masks_lattice + neg_rel_masks_lattice

    assert len(entity_masks) == len(entity_sizes) == len(entity_types)
    assert len(rels) == len(rel_masks) == len(rel_types)

    # create tensors
    # token indices
    encodings_length = torch.tensor(len(encodings), dtype=torch.long)
    encodings = torch.tensor(encodings, dtype=torch.long)
    lattice_encodings_length = torch.tensor(len(lattice_encodings), dtype=torch.long)
    lattice_encodings = torch.tensor(lattice_encodings, dtype=torch.long)

    # masking of tokens 上下文遮盖
    context_masks = torch.ones(context_size, dtype=torch.bool)
    context_masks_lattice = torch.ones(context_size_lattice, dtype=torch.bool)

    # also create samples_masks:
    # tensors to mask entity/relation samples of batch
    # since samples are stacked into batches, "padding" entities/relations possibly must be created
    # these are later masked during loss computation
    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        entity_masks = torch.stack(entity_masks)
        entity_masks_lattice = torch.stack(entity_masks_lattice)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg entities)
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_masks = torch.zeros([1, context_size_lattice], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    if rels:
        rels = torch.tensor(rels, dtype=torch.long)
        rel_masks = torch.stack(rel_masks)
        rel_masks_lattice = torch.stack(rel_masks_lattice)
        rel_types = torch.tensor(rel_types, dtype=torch.long)
        rel_sample_masks = torch.ones([rels.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg relations)
        rels = torch.zeros([1, 2], dtype=torch.long)
        rel_types = torch.zeros([1], dtype=torch.long)
        rel_masks = torch.zeros([1, context_size], dtype=torch.bool)
        rel_masks_lattice = torch.zeros([1, context_size_lattice], dtype=torch.bool)
        rel_sample_masks = torch.zeros([1], dtype=torch.bool)

    # relation types to one-hot encoding
    rel_types_onehot = torch.zeros([rel_types.shape[0], rel_type_count], dtype=torch.float32)

    rel_types_onehot.scatter_(1, rel_types.unsqueeze(1), 1)


    rel_types_onehot = rel_types_onehot[ :, 1:]  # all zeros for 'none' relation

    # 1.2构建pos的四个相对矩阵
    pos_e_tensor = torch.tensor(doc.pos_e, dtype=torch.int64)
    pos_s_tensor = torch.tensor(doc.pos_s, dtype=torch.int64)
    pos_ss_tensor = pos_s_tensor.unsqueeze(-1) - pos_s_tensor.unsqueeze(-2)
    pos_se_tensor = pos_s_tensor.unsqueeze(-1) - pos_e_tensor.unsqueeze(-2)
    pos_es_tensor = pos_e_tensor.unsqueeze(-1) - pos_s_tensor.unsqueeze(-2)
    pos_ee_tensor = pos_e_tensor.unsqueeze(-1) - pos_e_tensor.unsqueeze(-2)

    max_seq_length = pos_s_tensor.size(0)

    # 1.2.1进行pos合并
    pe_2_tensor = torch.cat([pos_ss_tensor,pos_ee_tensor],dim=-1)
    lex_num = len(doc.lex_s)
    lex_num = torch.tensor(lex_num,dtype=torch.int64)
    # print(doc.tokens)
    return dict(
                encodings=encodings, # 17
                encodings_length = encodings_length,
                lattice_encodings = lattice_encodings, # 24
                lattice_encodings_length = lattice_encodings_length,
                context_masks=context_masks,
                context_masks_lattice = context_masks_lattice,
                entity_masks=entity_masks,
                entity_masks_lattice=entity_masks_lattice,
                entity_sizes=entity_sizes,
                entity_types=entity_types,
                rels=rels,
                rel_masks=rel_masks,
                rel_masks_lattice=rel_masks_lattice,
                rel_types=rel_types_onehot,
                entity_sample_masks=entity_sample_masks,
                rel_sample_masks=rel_sample_masks,
                pos_ss_tensor=pos_ss_tensor,
                pos_se_tensor=pos_se_tensor,
                pos_es_tensor=pos_es_tensor,
                pos_ee_tensor=pos_ee_tensor,
                pos_e=pos_e_tensor,
                pos_s=pos_s_tensor,
                lex_num=lex_num
                )


def create_eval_sample(doc, max_span_size: int):
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)
    # 特殊构建的lattice机制嵌入
    lattice_encodings = doc.lattice_encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)
    context_size_lattice = len(lattice_encodings)
    # create entity candidates
    entity_spans = []
    entity_spans_lattice = []
    entity_masks = []
    entity_masks_lattice = []
    entity_sizes = []
    entity_sizes_lattice = []
    pos_entity_masks_lattice = []
    pos_entity_sizes_lattice = []

    # 原始长度序列的entity_mask
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            entity_spans.append(span)
            entity_masks.append(create_entity_mask(*span, context_size))    # 实体覆盖 长度需要为加上lattice的序列长度
            entity_sizes.append(size)

    # 增加了lattice的实体entity_mask
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            entity_spans_lattice.append(span)
            entity_masks_lattice.append(create_entity_mask(*span, context_size_lattice))    # 实体覆盖 长度需要为加上lattice的序列长度
            entity_sizes_lattice.append(size)

    # create tensors
    # token indices
    _encoding = encodings
    _encoding_lattice = lattice_encodings
    encodings = torch.zeros(context_size, dtype=torch.long)
    lattice_encodings = torch.tensor(lattice_encodings, dtype=torch.long)
    encodings[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)
    lattice_encodings[:len(_encoding_lattice)] = torch.tensor(_encoding_lattice, dtype=torch.long)

    # masking of tokens
    context_masks = torch.zeros(context_size, dtype=torch.bool)
    context_masks[:len(_encoding)] = 1  # context_mask遮盖有效部分 可测试遮盖前半部分
    context_masks_lattice = torch.zeros(context_size_lattice, dtype=torch.bool)
    context_masks_lattice[:len(lattice_encodings)] = 1

    # 1.2构建pos的四个相对矩阵
    pos_e_tensor = torch.tensor(doc.pos_e, dtype=torch.int64)
    pos_s_tensor = torch.tensor(doc.pos_s, dtype=torch.int64)
    # entities
    if entity_masks:
        entity_masks = torch.stack(entity_masks)
        entity_masks_lattice = torch.stack(entity_masks_lattice)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_sizes_lattice = torch.tensor(entity_sizes_lattice, dtype=torch.long)
        entity_spans = torch.tensor(entity_spans, dtype=torch.long)

        # tensors to mask entity samples of batch
        # since samples are stacked into batches, "padding" entities possibly must be created
        # these are later masked during evaluation
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
    else:
        # corner case handling (no entities)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    return dict(encodings=encodings,
                lattice_encodings=lattice_encodings,
                context_masks=context_masks,
                context_masks_lattice=context_masks_lattice,
                entity_masks=entity_masks,
                entity_masks_lattice=entity_masks_lattice,
                entity_sizes=entity_sizes,
                entity_sizes_lattice=entity_sizes_lattice,
                entity_spans=entity_spans,
                entity_sample_masks=entity_sample_masks,
                pos_e=pos_e_tensor,
                pos_s=pos_s_tensor
                )


def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)
    return mask


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch
