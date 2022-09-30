"""Select new instances given prediction and retrieval modules"""
import math
import collections
import torch
import numpy as np
from torchtext import data

from utils import scorer
from utils.torch_utils import example_to_dict, example_to_dict_bert

import torch.nn.functional as F

TOKEN = data.Field(sequential=True, batch_first=True, lower=True, include_lengths=True)
RELATION = data.Field(sequential=False, unk_token=None, pad_token=None)
POS = data.Field(sequential=True, batch_first=True)
NER = data.Field(sequential=True, batch_first=True)
PST = data.Field(sequential=True, batch_first=True)
PR_CONFIDENCE = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
SL_CONFIDENCE = data.Field(sequential=False, use_vocab=False, dtype=torch.float)

FIELDS = {
    "tokens": ("token", TOKEN),
    "stanford_pos": ("pos", POS),
    "ner": ("ner", NER),
    "relation": ("relation", RELATION),
    "subj_pst": ("subj_pst", PST),
    "obj_pst": ("obj_pst", PST),
    "pr_confidence": ("pr_confidence", PR_CONFIDENCE),
    "sl_confidence": ("sl_confidence", SL_CONFIDENCE),
}

def split_samples(dataset, meta_idxs, batch_size=20, conf_m1=None):
    """Split dataset using idxs

    Args:
        dataset (data.Dataset): Dataset instance
        meta_idxs (list): List of indexes with the form (idx, predict_label, gold_label)
        batch_size (int, optional): Defaults to 50
        conf_m1 (dict, optional): An optional attribute for confidence of samples for predictor
    """
    iterator_unlabeled = data.Iterator(
        dataset=dataset,
        batch_size=batch_size,
        repeat=False,
        train=False,
        shuffle=False,
        sort=True,
        sort_key=lambda x: -len(x.token),
        sort_within_batch=False,
    )
    examples = iterator_unlabeled.data()
    new_examples, rest_examples, example_ids = [], [], set(idx for idx, pred, actual in meta_idxs)
    if conf_m1 is not None:
        meta_idxs = [(meta_idxs[meta_i][0], meta_idxs[meta_i][1], meta_idxs[meta_i][2], conf_m1[meta_i][1], 1.0) 
            for meta_i in range(len(meta_idxs))]
    elif conf_m1 is None:
        meta_idxs = [(idx, pred, actual, 1.0, 1.0) for idx, pred, actual in meta_idxs]

    for idx, pred, _, m1_confidence, m2_confidence in meta_idxs:
        output = example_to_dict(examples[idx], m1_confidence, m2_confidence, pred)
        new_examples.append(data.Example.fromdict(output, FIELDS))
    rest_examples = [example for k, example in enumerate(examples) if k not in example_ids]
    return new_examples, rest_examples

def get_pseudo_label(dataset, meta_idxs):
    #
    iterator_unlabeled = data.Iterator(
            dataset=dataset,
            batch_size=20,
            repeat=False,
            train=False,
            shuffle=False,
            sort=True,
            sort_key=lambda x: -len(x.token),
            sort_within_batch=False
        )
    examples = iterator_unlabeled.data()
    new_examples, rest_examples, example_ids = [], [], set(idx for idx, _, _ in meta_idxs)
    new_examples = [example for k, example in enumerate(examples) if k in example_ids]
    rest_examples = [example for k, example in enumerate(examples) if k not in example_ids]
    return new_examples, rest_examples

def gen_pseudo_label(dataset, meta_idxs, label_idx=1, k_samples=0):
    examples = dataset.examples
    new_examples, rest_examples, example_ids = [], [], set(idx for idx, conf1, conf2, pseudo_label in meta_idxs)
    # new_examples = [example for k, example in enumerate(examples) if k in example_ids]
    for idx, conf1, conf2, pseudo_label in meta_idxs:
        if label_idx == 1:
            output = example_to_dict(examples[idx], conf1, conf2, pseudo_label)
        elif label_idx == 2:
            output = example_to_dict(examples[idx], conf2, conf1, pseudo_label)
        new_examples.append(data.Example.fromdict(output, FIELDS))
        k_samples -= 1
        if k_samples <= 0:
            break
    return new_examples

def select_samples(model_m1, model_m2, unlabeled_m1, unlabeled_m2, k_samples, opt):
    # predictor selection
    meta_idxs_m1, confidence_idxs_m1 = model_m1.retrieve(unlabeled_m1, len(unlabeled_m1))  # retrieve all the samples
    meta_idxs_m2, confidence_idxs_m2 = model_m2.retrieve(unlabeled_m2, len(unlabeled_m2))  # retrieve all the samples

    print("Infer on predictor: ")  # Track performance of predictor alone

    # for self-training
    if opt["integrate_method"] == "m1_only":
        return split_samples(unlabeled_m1, meta_idxs_m1[:k_samples])
    # for RE_Ensembling
    elif opt["integrate_method"] == "m2_only":
        return split_samples(unlabeled_m2, meta_idxs_m2[:k_samples])
    elif opt["integrate_method"] == "intersection":
        # select 2k个数据
        m1_exps, rest_m1_exps = split_samples(unlabeled_m1, meta_idxs_m1[:k_samples],
            conf_m1=confidence_idxs_m1)
        m2_exps, rest_m2_exps = split_samples(unlabeled_m2, meta_idxs_m2[:k_samples], 
            conf_m1=confidence_idxs_m2)
        # Cotraining's return
        # return m2_exps, m1_exps, rest_m1_exps, rest_m2_exps

        # CoT2
        m1_exps_ds = data.Dataset(m1_exps, fields=unlabeled_m1.fields)
        m2_exps_ds = data.Dataset(m2_exps, fields=unlabeled_m2.fields)
        # give them to the peer
        co_exps_m1, co_confidence_m1 = model_m2.retrieve(m1_exps_ds, len(m1_exps_ds))  # retrieve all the samples
        co_exps_m2, co_confidence_m2 = model_m1.retrieve(m2_exps_ds, len(m2_exps_ds))  # retrieve all the samples
        same_idx_m1 = []
        same_idx_m2 = []
        m1_sample = data.Iterator(
            dataset=m1_exps_ds,
            batch_size=20,
            repeat=False,
            train=False,
            shuffle=False,
            sort=True,
            sort_key=lambda x: -len(x.token),
            sort_within_batch=False
        ).data()
        m2_sample = data.Iterator(
            dataset=m2_exps_ds,
            batch_size=20,
            repeat=False,
            train=False,
            shuffle=False,
            sort=True,
            sort_key=lambda x: -len(x.token),
            sort_within_batch=False
        ).data()
        
        # calculate loss
        m1_loss = []
        for idx, conf, pred in co_confidence_m1:
            rel_id = opt['rel_stoi'][m1_sample[idx].relation]
            rel_conf = m1_sample[idx].pr_confidence
            x = torch.tensor(pred).view(1, -1)
            y = torch.tensor([rel_id])
            
            loss = rel_conf * F.cross_entropy(x, y)
            m1_loss.append([idx, loss.item()])
        
        m1_loss = sorted(m1_loss, key=lambda x: x[1])
        m1_loss = [m1_loss[i][0] for i in range(len(m1_loss)) if i < len(m1_loss) * 0.9]

        m2_loss = []
        for idx, conf, pred in co_confidence_m2:
            rel_id = opt['rel_stoi'][m2_sample[idx].relation]
            rel_conf = m2_sample[idx].pr_confidence
            x = torch.tensor(pred).view(1, -1)
            y = torch.tensor([rel_id])
            
            loss = rel_conf * F.cross_entropy(x, y)
            m2_loss.append([idx, loss.item()])
        
        m2_loss = sorted(m2_loss, key=lambda x: x[1])
        m2_loss = [m2_loss[i][0] for i in range(len(m2_loss)) if i < len(m2_loss) * 0.9]
        # get instances with same labels and small loss
        k = 0
        for idx, pred1_rel, pred2_rel in co_exps_m1:
            if pred1_rel == pred2_rel and idx in m1_loss:
                same_idx_m1.append((idx, pred1_rel, pred2_rel))
            else:
                m1_exps[idx].relation = meta_idxs_m1[idx][2]
            k += 1
        k = 0
        for idx, pred2_rel, pred1_rel in co_exps_m2:
            if pred2_rel == pred1_rel and idx in m2_loss:
                same_idx_m2.append((idx, pred2_rel, pred1_rel))
            else:
                m2_exps[idx].relation = meta_idxs_m2[idx][2]
            k += 1
        print("m1 added num : ", len(same_idx_m1))
        print("m2 added num : ", len(same_idx_m2))
        #
        new_m1_exps, _rest_m1_exps = get_pseudo_label(m1_exps_ds, same_idx_m1)
        new_m2_exps, _rest_m2_exps = get_pseudo_label(m2_exps_ds, same_idx_m2)
        # 剩余样本
        rest_m1_exps = rest_m1_exps + _rest_m1_exps
        rest_m2_exps = rest_m2_exps + _rest_m2_exps

        return new_m1_exps, new_m2_exps, rest_m1_exps, rest_m2_exps
    else:
        raise NotImplementedError("integrate_method {} not implemented".format(opt["integrate_method"]))
    
