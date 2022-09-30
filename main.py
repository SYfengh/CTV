import os
import time
import math
import random
import argparse

import torch
from torchtext import data

from selection import select_samples
from model.model1 import Model1
from model.trainer import Trainer, evaluate
from utils import helper, scorer, torch_utils

# 生成参数
def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset/semeval")
    # saved_models/semeval or saved_models/retacred 
    # PRNN saved_models/semeval/PRNN/model1
    parser.add_argument("--m1_dir", type=str, default="saved_models/semeval/CTV/0.15-0.5/seed1/model1", help="Directory of the model1")
    parser.add_argument("--m2_dir", type=str, default="saved_models/semeval/CTV/0.15-0.5/seed1/model2", help="Directory of the model2")

    parser.add_argument("--model_name", type=str, default="CTV") # model_name--PRNN/PRNN_gold/CoTraining/CTV
    parser.add_argument("--integrate_method", type=str, default="intersection") # intersection
    parser.add_argument("--num_iter", type=int, default=10) # 
    parser.add_argument("--labeled_ratio", type=float, default=0.05) # the ratio of labeled data--0.05/0.1/0.3/0.5
    parser.add_argument("--unlabeled_ratio", type=float, default=0.5) # the ratio of unlabeled data--only 0.5

    parser.add_argument("--word_dim", type=int, default=300) # word embedding dimension
    parser.add_argument("--ner_dim", type=int, default=30) # NER(entity) embedding dimension
    parser.add_argument("--pos_dim", type=int, default=30) # POS(part-of-speech) embedding dimension
    parser.add_argument("--pe_dim", type=int, default=30) # Position encoding dimension

    parser.add_argument("--hidden_dim", type=int, default=200) # RNN hidden state size
    parser.add_argument("--num_layers", type=int, default=2) # Num of RNN layers
    parser.add_argument("--attn", dest="attn", action="store_true", default=True) # PCNN/PRNN user attention layer
    parser.add_argument("--attn_dim", type=int, default=100) # PCNN/PRNN user attention size

    parser.add_argument("--dropout", type=float, default=0.5) # dropout rate
    parser.add_argument("--lr", type=float, default=1.0) # Applies to SGD and Adagrad
    parser.add_argument("--lr_decay", type=float, default=0.9) # 
    parser.add_argument("--optim", type=str, default="sgd") # sgd, adagrad, adam or adamax
    parser.add_argument("--num_epoch", type=int, default=30)
    parser.add_argument("--patience", type=int, default=0) # set default 0 and analyse its use
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=float, default=5.0) # Gradient clipping
    
    parser.add_argument("--log_step", type=int, default=100) # Print log every k steps
    parser.add_argument("--save_epoch", type=int, default=300) # Save model checkpoints every k epochs
    parser.add_argument("--save_dir", type=str, default="./saved_models") # Root dir for saving models

    parser.add_argument("--seed", type=int, default=1) 
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--device", type=int, default=1)

    args, _ = parser.parse_known_args()
    return args

def load_data(opt):
    print("Loading data from {} with batch size {}...".format(opt["data_dir"], opt["batch_size"]))
    TOKEN = data.Field(sequential=True, batch_first=True, lower=True, include_lengths=True)
    RELATION = data.Field(sequential=False, pad_token=None)
    POS = data.Field(sequential=True, batch_first=True)
    NER = data.Field(sequential=True, batch_first=True)
    PST = data.Field(sequential=True, batch_first=True)
    PR_CONFIDENCE = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
    SL_CONFIDENCE = data.Field(sequential=False, use_vocab=False, dtype=torch.float)

    FIELDS = {
        "tokens": ("token", TOKEN),
        "relation": ("relation", RELATION),
        "stanford_pos": ("pos", POS),
        "stanford_ner": ("ner", NER), # stanford_ner for retacred, ner for semeval
        "subj_pst": ("subj_pst", PST),
        "obj_pst": ("obj_pst", PST),
        "pr_confidence": ("pr_confidence", PR_CONFIDENCE),
        "sl_confidence": ("sl_confidence", SL_CONFIDENCE),
    }
    # all data are used for supervised learning
    #    
    # train_set = data.TabularDataset(path=opt["data_dir"] + "/train-" + str(opt["labeled_ratio"]) + ".json", format="json", fields=FIELDS)
    # train_set = data.TabularDataset(path=opt["data_dir"] + "/train.json", format="json", fields=FIELDS)
    # train_num = len(train_set.examples)
    # arr = np.arange(train_num)
    # np.random.shuffle(arr)
    # arr1 = arr[:int(train_num*0.1)] #
    # arr2 = arr[int(train_num*0.1):int(train_num*0.2)] #
    # train_set_m1 = data.Dataset(train_set.examples[int(train_num*0.1):], fields=train_set.fields)
    # train_set_m2 = data.Dataset(train_set.examples[:int(train_num*0.9)], fields=train_set.fields)

    # 随机生成两组数据
    train_set_m1 = data.TabularDataset(path=opt["data_dir"] + "/train-" + str(opt["labeled_ratio"]) + ".json", format="json", fields=FIELDS)
    train_set_m2 = data.TabularDataset(path=opt["data_dir"] + "/train-" + str(opt["labeled_ratio"]) + ".json", format="json", fields=FIELDS)
    dev_set = data.TabularDataset(path=opt["data_dir"] + "/dev.json", format="json", fields=FIELDS)
    test_set = data.TabularDataset(path=opt["data_dir"] + "/test.json", format="json", fields=FIELDS)
    unlabeled_set_m1 = data.TabularDataset(path=opt["data_dir"] + "/raw-" + str(opt["unlabeled_ratio"]) + ".json", format="json", fields=FIELDS)
    unlabeled_set_m2 = data.TabularDataset(path=opt["data_dir"] + "/raw-" + str(opt["unlabeled_ratio"]) + ".json", format="json", fields=FIELDS)

    print(
        "Labeled instances #: %d, Unlabeled instances #: %d" % (len(train_set_m1.examples), len(unlabeled_set_m1.examples))
    )
    dataset_vocab = data.TabularDataset(path=opt["data_dir"] + "/train.json", format="json", fields=FIELDS)
    TOKEN.build_vocab(dataset_vocab)
    RELATION.build_vocab(dataset_vocab)
    POS.build_vocab(dataset_vocab)
    NER.build_vocab(dataset_vocab)
    PST.build_vocab(dataset_vocab)
    opt["num_class"] = len(RELATION.vocab)
    opt["vocab_pad_id"] = TOKEN.vocab.stoi["<pad>"]
    opt["pos_pad_id"] = POS.vocab.stoi["<pad>"]
    opt["ner_pad_id"] = NER.vocab.stoi["<pad>"]
    opt["pe_pad_id"] = PST.vocab.stoi["<pad>"]

    opt["vocab_size"] = len(TOKEN.vocab)
    opt["pos_size"] = len(POS.vocab)
    opt["ner_size"] = len(NER.vocab)
    opt["pe_size"] = len(PST.vocab)

    opt["rel_stoi"] = RELATION.vocab.stoi
    opt["rel_itos"] = RELATION.vocab.itos

    TOKEN.vocab.load_vectors("glove.840B.300d", cache="glove/")
    if TOKEN.vocab.vectors is not None:
        opt["emb_dim"] = TOKEN.vocab.vectors.size(1)

    return train_set_m1, train_set_m2, dev_set, test_set, unlabeled_set_m1, unlabeled_set_m2, opt, TOKEN

def load_best_model(model_dir, model_type="m1"):
    model_file = model_dir + "/best_model.pt"
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    if model_type == "m1":
        m1 = Model1(model_opt)
        model = Trainer(model_opt, m1)
    else:
        m2 = Model1(model_opt)
        model = Trainer(model_opt, m2)
    model.load(model_file)
    helper.print_config(model_opt)
    return model

def best_m_pred(model, set, type):
    print("Final evaluation on " + type + " set...")
    return evaluate(model, set, verbose=True)
    

if __name__ == "__main__":
    args = init_parser()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # make opt
    opt = vars(args)

    train_set_m1, train_set_m2, dev_set, test_set, unlabeled_set_m1, unlabeled_set_m2, opt, TOKEN = load_data(opt)
    # unlabeled data for REGold model
    if opt["model_name"] == "PRNN_gold":
        exps = unlabeled_set_m1.examples
        for e in exps:
            e.pr_confidence = 1
            e.sl_confidence = 1
        train_set_m1 = data.Dataset(examples=train_set_m1.examples + exps, fields=train_set_m1.fields)


    num_iters = 11
    unlabeled_num = len(unlabeled_set_m1.examples)
    RT = 0.2
    k_samples = int(unlabeled_num * RT)

    dev_f1_iter, test_f1_iter = [], []
    # only train once for PRNN and REGold model
    if opt["model_name"] == "PRNN_gold" or opt["model_name"] == "PRNN":
        num_iters = 1
    helper.ensure_dir(opt["m1_dir"], verbose=True)
    if opt["model_name"] in ["CoTraining", "CTV"]:
        helper.ensure_dir(opt["m2_dir"], verbose=True)

    # model training
    for num_iter in range(num_iters):

        print("=" * 100)
        print("Training #: %d, Infer #: %d" % (len(train_set_m1.examples), len(unlabeled_set_m1.examples)))
        opt["model_save_dir"] = opt["m1_dir"]
        # save config
        helper.save_config(opt, opt["model_save_dir"] + "/config.json", verbose=True)
        helper.print_config(opt)

        # model1
        if num_iter == 0:
            model1 = Model1(opt, emb_matrix=TOKEN.vocab.vectors)
            model = Trainer(opt, model1) 
        else:
            model = load_best_model(opt["model_save_dir"])
        model.train(train_set_m1, dev_set)

        # 加载存储的最好模型
        best_model_m1 = load_best_model(opt["model_save_dir"])
        _ = best_m_pred(best_model_m1, train_set_m1, "train")[2]
        dev_f1 = best_m_pred(best_model_m1, dev_set, "dev")[2]
        test_f1 = best_m_pred(best_model_m1, test_set, "test")[2]

        dev_f1_iter.append(dev_f1)
        test_f1_iter.append(test_f1)
        best_model_m1 = load_best_model(opt["model_save_dir"])

        # CoTraining with model2
        if opt["model_name"] in ["CoTraining", "CTV"]:
            opt["model_save_dir"] = opt["m2_dir"]
            helper.save_config(opt, opt["model_save_dir"] + "/config.json", verbose=True)
            helper.print_config(opt)
            if num_iter == 0:
                model2 = Model1(opt, emb_matrix=TOKEN.vocab.vectors)
                model = Trainer(opt, model2)
            else:
                model = load_best_model(opt["model_save_dir"])
            model.train(train_set_m2, dev_set)

            # 加载存储的最好模型
            best_model_m2 = load_best_model(opt["model_save_dir"])
            _ = best_m_pred(best_model_m2, train_set_m2, "train")[2]
            dev_f1 = best_m_pred(best_model_m2, dev_set, "dev")[2]
            test_f1 = best_m_pred(best_model_m2, test_set, "test")[2]

            dev_f1_iter.append(dev_f1)
            test_f1_iter.append(test_f1)
            best_model_m2 = load_best_model(opt["model_save_dir"])
 
            new_m1_exps, new_m2_exps, rest_m1_exps, rest_m2_exps = select_samples(
                best_model_m1, best_model_m2, unlabeled_set_m1, unlabeled_set_m2, k_samples, opt
            )

            train_set_m1.examples = train_set_m1.examples + new_m1_exps
            train_set_m2.examples = train_set_m2.examples + new_m2_exps
            unlabeled_set_m1.examples = rest_m1_exps
            unlabeled_set_m2.examples = rest_m2_exps
    scorer.print_table(dev_f1_iter, test_f1_iter, header="Best dev and test F1 with seed=%s:" % args.seed)
    # final select
