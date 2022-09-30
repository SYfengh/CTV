"""
Utility functions for torch.
"""
import torch
from torch.optim import Optimizer
import numpy as np

# torch specific functions
def get_optimizer(name, parameters, lr):
    name = name.strip()
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    elif name == "adam":
        return torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.99))  # use default lr
    elif name == "adamax":
        return torch.optim.Adamax(parameters)  # use default lr
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def flatten_indices(seq_lens, width):
    flat = []
    for i, l in enumerate(seq_lens):
        for j in range(l):
            flat.append(i * width + j)
    return flat

def keep_partial_grad(grad, topk):
    """
    Keep only the topk rows of grads.
    """
    assert topk < grad.size(0)
    grad.data[topk:].zero_()
    return grad

def unsort_idx(examples, batch_size):
    def unsort(lengths):
        return sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True)

    lengths = np.array([len(ex.token) for ex in examples])
    # idxs = [np.argsort(np.argsort(- lengths[i:i+batch_size])) for i in range(0, len(lengths), batch_size)]
    idxs = [np.argsort(unsort(lengths[i : i + batch_size])) for i in range(0, len(lengths), batch_size)]
    return [torch.LongTensor(idx) for idx in idxs]


# model IO
def save(model, optimizer, opt, filename):
    params = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": opt}
    try:
        torch.save(params, filename)
    except BaseException:
        print("[ Warning: model saving failed. ]")


def load(model, optimizer, filename):
    try:
        dump = torch.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    if model is not None:
        model.load_state_dict(dump["model"])
    if optimizer is not None:
        optimizer.load_state_dict(dump["optimizer"])
    opt = dump["config"]
    return model, optimizer, opt


def load_config(filename):
    dump = torch.load(filename)
    return dump["config"]


# data batch
def batch_to_input(batch, vocab_pad_id=0):
    inputs = {}
    inputs["words"], inputs["length"] = batch.token
    inputs["pos"] = batch.pos
    inputs["ner"] = batch.ner
    inputs["subj_pst"] = batch.subj_pst
    inputs["obj_pst"] = batch.obj_pst
    inputs["masks"] = torch.eq(batch.token[0], vocab_pad_id)
    inputs["pr_confidence"] = batch.pr_confidence
    inputs["sl_confidence"] = batch.sl_confidence
    if inputs["pos"].shape[0] != inputs["words"].shape[0] or inputs["pos"].shape[1] != inputs["words"].shape[1]:
        print(batch.token)
    return inputs, batch.relation

def batch_to_input_bert(batch):
    inputs = {}
    inputs["words"], inputs["length"] = batch.token
    inputs["pos"] = batch.pos
    inputs["ner"] = batch.ner
    inputs["subj_start"] = batch.subj_start
    inputs["obj_start"] = batch.obj_start
    inputs["masks"] = batch.mask
    inputs["pr_confidence"] = batch.pr_confidence
    inputs["sl_confidence"] = batch.sl_confidence
    if inputs["pos"].shape[0] != inputs["words"].shape[0] or inputs["pos"].shape[1] != inputs["words"].shape[1]:
        print(batch.token)
    return inputs, batch.relation


def example_to_dict(example, pr_confidence, sl_confidence, rel):
    output = {}
    output["tokens"] = example.token
    output["stanford_pos"] = example.pos
    output["ner"] = example.ner
    output["subj_pst"] = example.subj_pst
    output["obj_pst"] = example.obj_pst
    output["relation"] = rel
    output["pr_confidence"] = pr_confidence
    output["sl_confidence"] = sl_confidence
    return output

def example_to_dict_bert(example, pr_confidence, sl_confidence, rel):
    output = {}
    output["tokens"] = example.token
    output["pos"] = example.pos
    output["ner"] = example.ner
    output["mask"] = example.mask
    output["subj_start"] = example.subj_start
    output["obj_start"] = example.obj_start
    output["relation"] = rel
    output["pr_confidence"] = pr_confidence
    output["sl_confidence"] = sl_confidence

    return output

def arg_max(l):
    bvl, bid = -1, -1
    for k in range(len(l)):
        if l[k] > bvl:
            bvl = l[k]
            bid = k
    return bid, bvl
