from abc import ABC

import torch


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass

"""这是用来计算模型整体损失的类和方法"""
class cival_wospert_Loss(Loss):
    def __init__(self, rel_criterion, entity_criterion, model, optimizer, scheduler, max_grad_norm):
        self._rel_criterion = rel_criterion
        self._entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm

    def compute(self, entity_logits, rel_logits, entity_types, rel_types, entity_sample_masks, rel_sample_masks):
        # entity loss
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()

        entity_loss = self._entity_criterion(entity_logits, entity_types)
        entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

        # relation loss
        rel_sample_masks = rel_sample_masks.view(-1).float()
        rel_count = rel_sample_masks.sum()

        if rel_count.item() != 0:
            rel_logits = rel_logits.view(-1, rel_logits.shape[-1])
            rel_types = rel_types.view(-1, rel_types.shape[-1])

            rel_loss = self._rel_criterion(rel_logits, rel_types)
            rel_loss = rel_loss.sum(-1) / rel_loss.shape[-1]
            rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count

            # joint loss
            train_loss = entity_loss + rel_loss
        else:
            # corner case: no positive/negative relation samples
            train_loss = entity_loss

        print(train_loss)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()


    def compute_entity(self, entity_logits, entity_types, entity_sample_masks):
        # entity loss
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()

        entity_loss = self._entity_criterion(entity_logits, entity_types)
        entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

        train_loss = entity_loss

        print(train_loss)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()
