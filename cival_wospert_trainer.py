import argparse
import math
import os

import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
import transformers
from torch.utils.data import DataLoader
from transformers import AdamW, BertConfig
from transformers import BertTokenizer

from cival_wospert import models
from cival_wospert import sampling
from cival_wospert import util
from cival_wospert.entities_rules import Dataset
from cival_wospert.evaluator import Evaluator
from cival_wospert.input_reader import JsonInputReader, BaseInputReader
from cival_wospert.loss import cival_wospert_Loss, Loss
from tqdm import tqdm

from cival_wospert.models import cival_wospert
from cival_wospert.trainer import BaseTrainer

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class Cival_woSpert(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # byte-pair encoding
        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                        do_lower_case=args.lowercase,
                                                        cache_dir=args.cache_path)

        # path to export predictions to
        self._predictions_path = os.path.join(self._log_path, 'predictions_%s_epoch_%s.json')

        # path to export relation extraction examples to
        self._examples_path = os.path.join(self._log_path, 'examples_%s_%s_epoch_%s.html')

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        train_label, valid_label = 'train', 'valid'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("Model type: %s" % args.model_type)

        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # read datasets 读取数据 训练集和验证集的数据 并保存在input_reader._datasets中 对于conll04 训练集实体有3377个，关系有1283个
        input_reader = input_reader_cls(types_path, self._tokenizer, args.neg_entity_count,
                                        args.neg_relation_count, args.max_span_size, self._logger)
        input_reader.read({train_label: train_path, valid_label: valid_path})
        self._log_datasets(input_reader)    # 吧这些读取到的数据信息log存储起来

        train_dataset = input_reader.get_dataset(train_label) # 得到训练集的数据
        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // args.train_batch_size # 总样本数 ÷ 训练的batch大小数 = 单个epoch的迭代次数
        updates_total = updates_epoch * args.epochs # 在需要迭代的epoch次数情况下需要的迭代次数

        validation_dataset = input_reader.get_dataset(valid_label)  # 得到验证集的数据

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        # create model 创建模型 默认为cival_wospert模型
        model_class = models.get_model(self.args.model_type)

        # load model
        config = BertConfig.from_pretrained(self.args.model_path, cache_dir=self.args.cache_path)
        util.check_version(config, model_class, self.args.model_path)

        config.cival_wospert_version = model_class.VERSION
        # 模型初始化 初始化各个层
        model = model_class.from_pretrained(self.args.model_path,
                                            config=config,
                                            # Cival_wospert model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            relation_types=input_reader.relation_type_count-1,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer)
        model.resize_token_embeddings(len(self._tokenizer))
        print("重新调整Bert模型词典大小为:" + str(len(self._tokenizer)))

        model.to(self._device)

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        # create scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total)
        # create loss function
        rel_criterion = torch.nn.BCEWithLogitsLoss(reduction='none') # 创建loss函数 关系分类采用了BCE loss
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none') # 创建loss函数 实体分类采用交叉熵损失
        compute_loss = cival_wospert_Loss(rel_criterion, entity_criterion, model, optimizer, scheduler, args.max_grad_norm)

        # eval validation set
        if args.init_eval:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)

        # train
        for epoch in range(args.epochs):
            # train epoch
            self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch)


            # eval validation sets
            if not args.final_eval or (epoch == args.epochs - 1):
                print("eval in: " + str(epoch) + " epoch")
                self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)

        # save final model
        extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
        global_iteration = args.epochs * updates_epoch
        # save final model_bert
        # modelBertBlockModel = Cival_woSpert.get_bert_block(model)
        # modelBertAttenModel = Cival_woSpert.get_att_block(model)
        # 保存整个模型 还是只保存BERT
        self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                         optimizer=optimizer if self.args.save_optimizer else None, extra=extra,
                         include_iteration=False, name='final_model')
        # self._save_model_bert_block(self._save_path, model, self._tokenizer, global_iteration,
        #                  optimizer=optimizer if self.args.save_optimizer else None, extra=None,
        #                  include_iteration=False, name='final_model_bert_block')
        # 保存预训练的模型
        # modelBertBlockModel.save_pretrained(self._save_path_bert_block)
        # state_path = os.path.join(self._save_path_bert_block, 'extra_att.state')
        # extra_state_atten = modelBertAttenModel.state_dict()
        # torch.save(extra_state_atten, state_path)

        self._logger.info("Logged in: %s" % self._log_path)
        self._logger.info("Saved in: %s" % self._save_path)
        self._close_summary_writer()

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer,
                                        max_span_size=args.max_span_size, logger=self._logger)
        input_reader.read({dataset_label: dataset_path})
        self._log_datasets(input_reader)

        # create model
        model_class = models.get_model(self.args.model_type)

        config = BertConfig.from_pretrained(self.args.model_path, cache_dir=self.args.cache_path)
        util.check_version(config, model_class, self.args.model_path)

        # checkpoint = torch.load('E:\\NLP\\NER_RE\\spert-master\\scripts\\data\\save\\rules_datasets_train\\tempDir\\final_model\\extra.state')
        model = model_class.from_pretrained(
                                            self.args.model_path,
                                            config=config,
                                            # cival_wospert model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            relation_types=input_reader.relation_type_count-1,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer)

        # model.load_state_dict(checkpoint['optimizer']['state'])
        model.to(self._device)


        # evaluate
        self._eval(model, input_reader.get_dataset(dataset_label), input_reader)

        self._logger.info("Logged in: %s" % self._log_path)
        self._close_summary_writer()
    # 训练模型的方法 传入模型、优化器、损失计算器、数据集
    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int):
        self._logger.info("Train epoch: %s" % epoch)

        # create data loader
        dataset.switch_mode(Dataset.TRAIN_MODE) # 设置dataset的训练模式 默认选择train
        data_loader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=False, drop_last=True,
                                 num_workers=0, collate_fn=sampling.collate_fn_padding)

        model.zero_grad()

        iteration = 0
        total = dataset.document_count // self.args.train_batch_size
        count_index = 0
        # 选择任务类别!!!
        taskList = ['entity_relation', 'entity']
        task_name = taskList[0]
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            try:
                model.train()
                batch = util.to_device(batch, self._device)

                if task_name == 'entity_relation':
                    # forward step
                    # 1.实体和关系同时抽取
                    entity_logits, rel_logits = model(
                                                      task_name=task_name,
                                                      encodings=batch['encodings'],
                                                      encodings_length=batch['encodings_length'],
                                                      lattice_encodings=batch['lattice_encodings'],
                                                      lattice_encodings_length=batch['lattice_encodings_length'],
                                                      context_masks=batch['context_masks'],
                                                      context_masks_lattice=batch['context_masks_lattice'],
                                                      entity_masks=batch['entity_masks'],
                                                      entity_masks_lattice=batch['entity_masks_lattice'],
                                                      entity_sizes=batch['entity_sizes'],
                                                      relations=batch['rels'],
                                                      rel_masks=batch['rel_masks'],
                                                      rel_masks_lattice=batch['rel_masks_lattice'],
                                                      pos_ss_tensor=batch['pos_ss_tensor'],
                                                      pos_se_tensor=batch['pos_se_tensor'],
                                                      pos_es_tensor=batch['pos_es_tensor'],
                                                      pos_ee_tensor=batch['pos_ee_tensor'],
                                                      pos_e=batch['pos_e'],
                                                      pos_s=batch['pos_s'],
                                                      lex_num=batch['lex_num']
                                                        )
                    # compute loss and optimize parameters
                    batch_loss = compute_loss.compute(entity_logits=entity_logits,
                                                      rel_logits=rel_logits,
                                                      rel_types=batch['rel_types'],
                                                      entity_types=batch['entity_types'],
                                                      entity_sample_masks=batch['entity_sample_masks'],
                                                      rel_sample_masks=batch['rel_sample_masks'])
                elif task_name == 'entity':
                    # 2.只进行实体抽取
                    entity_logits = model(
                                                      task_name=task_name,
                                                      encodings=batch['encodings'],
                                                      encodings_length=batch['encodings_length'],
                                                      lattice_encodings=batch['lattice_encodings'],
                                                      lattice_encodings_length=batch['lattice_encodings_length'],
                                                      context_masks=batch['context_masks'],
                                                      context_masks_lattice=batch['context_masks_lattice'],
                                                      entity_masks=batch['entity_masks'],
                                                      entity_masks_lattice=batch['entity_masks_lattice'],
                                                      entity_sizes=batch['entity_sizes'],
                                                      relations=batch['rels'],
                                                      rel_masks=batch['rel_masks'],
                                                      rel_masks_lattice=batch['rel_masks_lattice'],
                                                      pos_ss_tensor=batch['pos_ss_tensor'],
                                                      pos_se_tensor=batch['pos_se_tensor'],
                                                      pos_es_tensor=batch['pos_es_tensor'],
                                                      pos_ee_tensor=batch['pos_ee_tensor'],
                                                      pos_e=batch['pos_e'],
                                                      pos_s=batch['pos_s'],
                                                      lex_num=batch['lex_num']
                                                        )
                    # compute loss and optimize parameters
                    batch_loss = compute_loss.compute_entity(entity_logits=entity_logits,
                                                      entity_types=batch['entity_types'],
                                                      entity_sample_masks=batch['entity_sample_masks'])

                # logging
                iteration += 1
                global_iteration = epoch * updates_epoch + iteration

                if global_iteration % self.args.train_log_iter == 0:
                    self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)
            except Exception as e:
                print(e)


        return iteration

    def _eval(self, model: torch.nn.Module, dataset: Dataset, input_reader: JsonInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        self._logger.info("Evaluate: %s" % dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator
        evaluator = Evaluator(dataset, input_reader, self._tokenizer,
                              self.args.rel_filter_threshold, self.args.no_overlapping, self._predictions_path,
                              self._examples_path, self.args.example_count, epoch, dataset.label)

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=0, collate_fn=sampling.collate_fn_padding)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self.args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                try:
                    # move batch to selected device
                    batch = util.to_device(batch, self._device)
                    # 跳过过长的序列
                    # if len(batch['encodings']) > 400:
                    #     print('a')
                    #     continue
                    # run model (forward pass)
                    result = model(encodings=batch['encodings'],
                                   lattice_encodings=batch['lattice_encodings'],
                                   context_masks=batch['context_masks'],
                                   context_masks_lattice=batch['context_masks_lattice'],
                                   entity_masks=batch['entity_masks'],
                                   entity_masks_lattice=batch['entity_masks_lattice'],
                                   entity_sizes=batch['entity_sizes'],
                                   entity_sizes_lattice=batch['entity_sizes_lattice'],
                                   entity_spans=batch['entity_spans'],
                                   entity_sample_masks=batch['entity_sample_masks'],
                                   pos_e=batch['pos_e'],
                                   pos_s=batch['pos_s'],
                                   evaluate=True)
                    entity_clf, rel_clf, rels = result
                except Exception as e:
                    print(e)
                    continue
                # evaluate batch
                evaluator.eval_batch(entity_clf, rel_clf, rels, batch)



        global_iteration = epoch * updates_epoch + iteration
        ner_eval, rel_eval, rel_nec_eval = evaluator.compute_scores()
        self._log_eval(*ner_eval, *rel_eval, *rel_nec_eval,
                       epoch, iteration, global_iteration, dataset.label)

        if self.args.store_predictions and not self.args.no_overlapping:
            evaluator.store_predictions()

        if self.args.store_examples:
            evaluator.store_examples()

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self.args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,

                  rel_prec_micro: float, rel_rec_micro: float, rel_f1_micro: float,
                  rel_prec_macro: float, rel_rec_macro: float, rel_f1_macro: float,

                  rel_nec_prec_micro: float, rel_nec_rec_micro: float, rel_nec_f1_micro: float,
                  rel_nec_prec_macro: float, rel_nec_rec_macro: float, rel_nec_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_prec_micro', rel_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_micro', rel_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_micro', rel_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_prec_macro', rel_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_macro', rel_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_macro', rel_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_nec_prec_micro', rel_nec_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_micro', rel_nec_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_micro', rel_nec_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_prec_macro', rel_nec_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_macro', rel_nec_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_macro', rel_nec_f1_macro, global_iteration)

        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,

                      rel_prec_micro, rel_rec_micro, rel_f1_micro,
                      rel_prec_macro, rel_rec_macro, rel_f1_macro,

                      rel_nec_prec_micro, rel_nec_rec_micro, rel_nec_f1_micro,
                      rel_nec_prec_macro, rel_nec_rec_macro, rel_nec_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        self._logger.info("Relations:")
        for r in input_reader.relation_types.values():
            self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

        self._logger.info("Context size: %s" % input_reader.context_size)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro',
                                                 'rel_prec_macro', 'rel_rec_macro', 'rel_f1_macro',
                                                 'rel_nec_prec_micro', 'rel_nec_rec_micro', 'rel_nec_f1_micro',
                                                 'rel_nec_prec_macro', 'rel_nec_rec_macro', 'rel_nec_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})
