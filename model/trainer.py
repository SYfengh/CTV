"""
A rnn model for relation extraction, written in pytorch.
"""
import math
import time
import os
from datetime import datetime
from shutil import copyfile
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchtext import data

from utils import torch_utils, scorer
from utils.torch_utils import batch_to_input, arg_max, batch_to_input_bert


def evaluate(model, dataset, verbose=False):
    rel_stoi, rel_itos = model.opt['rel_stoi'], model.opt['rel_itos']
    iterator_test = data.Iterator(
        dataset=dataset,
        batch_size=model.opt['batch_size'],
        repeat=False,
        train=True,
        shuffle=False,
        sort=True,
        sort_key=lambda x: -len(x.token),
        sort_within_batch=False)
    predictions = []
    all_probs = []
    golds = []
    all_loss = 0
    for batch in iterator_test:
        inputs, target = batch_to_input(batch, model.opt['vocab_pad_id'])
        preds, probs, loss = model.predict(inputs, target)
        predictions += preds
        all_probs += probs
        all_loss += loss
        golds += target.data.tolist()
    predictions = [rel_itos[p] for p in predictions]
    golds = [rel_itos[p] for p in golds]
    p, r, f1 = scorer.score(golds, predictions, verbose=verbose)
    return p, r, f1, all_loss


def calc_confidence(probs, exp):
    '''Calculate confidence score from raw probabilities'''
    return max(probs)**exp


class Trainer(object):
    """ A wrapper class for the training and evaluation of models. """

    def __init__(self, opt, model):
        self.opt = opt
        self.model = model
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]

        if opt['cuda']:
            self.model.cuda(opt['device'])
            self.criterion.cuda(opt['device'])

        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def train(self, dataset_train, dataset_dev):
        opt = self.opt.copy()
        iterator_train = data.Iterator(
            dataset=dataset_train,
            batch_size=opt['batch_size'],
            repeat=False,
            train=True,
            shuffle=True,
            sort_key=lambda x: len(x.token),
            sort_within_batch=True)
        iterator_dev = data.Iterator(
            dataset=dataset_dev,
            batch_size=opt['batch_size'],
            repeat=False,
            train=True,
            sort_key=lambda x: len(x.token),
            sort_within_batch=True)
        dev_score_history = []
        current_lr = opt['lr']

        global_step = 0
        format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
        max_steps = len(iterator_train) * opt['num_epoch']

        # start training
        epoch = 0
        patience = 0
        while True:
            epoch = epoch + 1
            train_loss = 0

            for batch in iterator_train:
                start_time = time.time()
                global_step += 1

                inputs, target = batch_to_input(batch, opt['vocab_pad_id'])
                loss = self.update(inputs, target)
                train_loss += loss
                if global_step % opt['log_step'] == 0:
                    duration = time.time() - start_time
                    print(
                        format_str.format(datetime.now(), global_step, max_steps, epoch,
                                          opt['num_epoch'], loss, duration, current_lr))

            # eval on dev
            print("Evaluating on dev set...")
            dev_p, dev_r, dev_score, dev_loss = evaluate(self, dataset_dev)

            # print training information
            train_loss = train_loss / len(iterator_train) * opt['batch_size']  # avg loss per batch
            dev_loss = dev_loss / len(iterator_dev) * opt['batch_size']
            print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_score = {:.4f}".format(
                epoch, train_loss, dev_loss, dev_score))

            # save the current model
            model_file = opt['model_save_dir'] + '/checkpoint_epoch_{}.pt'.format(epoch)
            # bert先不保存模型

            self.save(model_file, epoch)

            # 不考虑epoch为1的时候是否更好，结果显示，在新加入数据之后保留fine-tune的模型结果要更好一点
            if epoch == 1 or dev_score > max(dev_score_history):  # new best
                copyfile(model_file, opt['model_save_dir'] + '/best_model.pt')
                print("new best model saved.")
                patience = 0
            else:
                patience = patience + 1
            if epoch % opt['save_epoch'] != 0:
                os.remove(model_file)
                
            # change learning rate
            if len(dev_score_history) > 10 and dev_score <= dev_score_history[-1] and \
                    opt['optim'] in ['sgd', 'adagrad']:
                current_lr *= opt['lr_decay']
                self.update_lr(current_lr)

            dev_score_history += [dev_score]
            print("")
            if opt['patience'] != 0:
                if patience == opt['patience'] and epoch > opt['num_epoch']:
                    break
            else:
                if epoch == opt['num_epoch']:
                    break
        print("Training ended with {} epochs.".format(epoch))

    def retrieve(self, dataset, k_samples):
        iterator_unlabeled = data.Iterator(
            dataset=dataset,
            batch_size=self.opt['batch_size'],
            repeat=False,
            train=False,
            shuffle=False,
            sort=True,
            sort_key=lambda x: -len(x.token),
            sort_within_batch=False
        )

        preds = []

        for batch in iterator_unlabeled:
            inputs, _ = batch_to_input(batch, self.opt['vocab_pad_id'])
            preds += self.predict(inputs)[1]

        meta_idxs = []
        confidence_idxs = []
        examples = iterator_unlabeled.data()
        num_instance = len(examples)

        # ranking
        ranking = list(zip(range(num_instance), preds))
        ranking = sorted(
            ranking, key=lambda x: calc_confidence(x[1], self.opt['alpha']), reverse=True)
        # selection
        for eid, pred in ranking:
            if len(meta_idxs) == k_samples:
                break
            rid, _ = arg_max(pred)
            val = calc_confidence(pred, self.opt['alpha'])
            rel = self.opt['rel_itos'][rid]

            meta_idxs.append((eid, rel, examples[eid].relation))
            confidence_idxs.append((eid, val, pred))
        return meta_idxs, confidence_idxs

    # train the model with a batch
    def update(self, inputs, target):
        """ Run a step of forward and backward model update. """
        self.model.train()
        self.optimizer.zero_grad()

        sl_confidence = inputs['sl_confidence']

        if self.opt['cuda']:
            target = target.cuda(self.opt['device'])
            inputs = dict([(k, v.cuda(self.opt['device'])) for k, v in inputs.items()])
            pr_confidence = inputs['pr_confidence']

        logits, _ = self.model(inputs)

        loss = self.criterion(logits, target)
        loss = torch.mean(loss * pr_confidence)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.item()
        return loss_val

    def predict(self, inputs, target=None):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        if self.opt['cuda']:
            inputs = dict([(k, v.cuda(self.opt['device'])) for k, v in inputs.items()])
            target = None if target is None else target.cuda(self.opt['device'])

        self.model.eval()
        logits, _ = self.model(inputs)
        loss = None if target is None else torch.mean(self.criterion(logits, target)).item()

        probs = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        predictions = np.argmax(probs, axis=1).tolist()

        return predictions, probs, loss

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    # save the model
    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),  # model parameters
            'encoder': self.model.encoder.state_dict(),
            'classifier': self.model.classifier.state_dict(),
            'config': self.opt,  # options
            'epoch': epoch,  # current epoch
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    # load the model
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.encoder.load_state_dict(checkpoint['encoder'])
        self.model.classifier.load_state_dict(checkpoint['classifier'])
        self.opt = checkpoint['config']
        self.criterion = nn.CrossEntropyLoss()
