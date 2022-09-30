"""
A rnn model for relation extraction, written in pytorch.
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertTokenizer
from torch.nn import CrossEntropyLoss, MSELoss

class PositionAwareAttention(nn.Module):
    """
    A position-augmented attention layer where the attention weight is
    a = T' . tanh(Ux + Vq + Wf)
    where x is the input, q is the query, and f is additional position features.
    """

    def __init__(self, input_size, query_size, feature_size, attn_size):
        super(PositionAwareAttention, self).__init__()
        self.input_size = input_size
        self.query_size = query_size
        self.feature_size = feature_size
        self.attn_size = attn_size
        self.ulinear = nn.Linear(input_size, attn_size)
        self.vlinear = nn.Linear(query_size, attn_size, bias=False)
        if feature_size > 0:
            self.wlinear = nn.Linear(feature_size, attn_size, bias=False)
        else:
            self.wlinear = None
        self.tlinear = nn.Linear(attn_size, 1)
        self.init_weights()

    def init_weights(self):
        self.ulinear.weight.data.normal_(std=0.001)
        self.vlinear.weight.data.normal_(std=0.001)
        if self.wlinear is not None:
            self.wlinear.weight.data.normal_(std=0.001)
        self.tlinear.weight.data.zero_()  # use zero to give uniform attention at the beginning

    def forward(self, x, x_mask, q, f):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        f : batch_size * seq_len * feature_size

        x is the sequence of word embeddings
        q is the last hidden state
        f is the position embeddings
        """
        batch_size, seq_len, _ = x.size()

        x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
            batch_size, seq_len, self.attn_size
        )
        q_proj = (
            self.vlinear(q.view(-1, self.query_size))
            .contiguous()
            .view(batch_size, self.attn_size)
            .unsqueeze(1)
            .expand(batch_size, seq_len, self.attn_size)
        )
        if self.wlinear is not None:
            f_proj = (
                self.wlinear(f.view(-1, self.feature_size))
                .contiguous()
                .view(batch_size, seq_len, self.attn_size)
            )
            projs = [x_proj, q_proj, f_proj]
        else:
            projs = [x_proj, q_proj]
        scores = self.tlinear(torch.tanh(sum(projs)).view(-1, self.attn_size)).view(
            batch_size, seq_len
        )

        # mask padding
        scores.data.masked_fill_(x_mask.data, -float("inf"))
        weights = F.softmax(scores, dim=-1)
        # weighted average input vectors
        outputs = weights.unsqueeze(1).bmm(x).squeeze(1)
        return outputs


class RNNEncoder(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(RNNEncoder, self).__init__()
        self.drop = nn.Dropout(opt["dropout"])
        self.emb = nn.Embedding(
            opt["vocab_size"], opt["emb_dim"], padding_idx=opt["vocab_pad_id"]
        )
        if opt["pos_dim"] > 0:
            self.pos_emb = nn.Embedding(
                opt["pos_size"], opt["pos_dim"], padding_idx=opt["pos_pad_id"]
            )
        if opt["ner_dim"] > 0:
            self.ner_emb = nn.Embedding(
                opt["ner_size"], opt["ner_dim"], padding_idx=opt["ner_pad_id"]
            )

        input_size = opt["emb_dim"] + opt["pos_dim"] + opt["ner_dim"]
        self.rnn = nn.LSTM(
            input_size,
            opt["hidden_dim"],
            opt["num_layers"],
            batch_first=True,
            dropout=opt["dropout"],
        )

        # attention layer
        if opt["attn"]:
            self.attn_layer = PositionAwareAttention(
                opt["hidden_dim"], opt["hidden_dim"], 2 * opt["pe_dim"], opt["attn_dim"]
            )
            self.pe_emb = nn.Embedding(
                opt["pe_size"], opt["pe_dim"], padding_idx=opt["pe_pad_id"]
            )

        self.opt = opt
        self.use_cuda = opt["cuda"]
        self.emb_matrix = emb_matrix
        if emb_matrix is not None:
            self.emb.weight.data.copy_(emb_matrix)

    def zero_state(self, batch_size):
        state_shape = (self.opt["num_layers"], batch_size, self.opt["hidden_dim"])
        h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
        if self.use_cuda:
            return h0.cuda(self.opt['device']), c0.cuda(self.opt['device'])
        else:
            return h0, c0

    def forward(self, inputs):
        # words: [batch size, seq length]
        seq_lens = inputs["length"]
        words, masks = inputs["words"], inputs["masks"]
        pos, ner = inputs["pos"], inputs["ner"]
        len = pos.shape[1]
        subj_pst, obj_pst = inputs["subj_pst"][:,:len], inputs["obj_pst"][:,:len]
        # 

        batch_size = words.size()[0]

        # embedding lookup
        # word_inputs: [batch size, seq length, embedding size]
        # inputs: [batch size, seq length, embedding size * 3]
        word_inputs = self.emb(words)
        inputs = [word_inputs]
        if self.opt["pos_dim"] > 0:
            inputs += [self.pos_emb(pos)]
        if self.opt["ner_dim"] > 0:
            inputs += [self.ner_emb(ner)]
        inputs = self.drop(torch.cat(inputs, dim=2))  # add dropout to input

        # rnn
        h0, c0 = self.zero_state(batch_size)
        inputs = nn.utils.rnn.pack_padded_sequence(
            inputs, seq_lens.tolist(), batch_first=True
        )
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        outputs, output_lens = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True
        )
        hidden = self.drop(ht[-1, :, :])  # get the outmost layer h_n
        outputs = self.drop(outputs)

        # attention
        if self.opt["attn"]:
            # convert all negative PE numbers to positive indices
            # e.g., -2 -1 0 1 will be mapped to 98 99 100 101
            subj_pe_inputs = self.pe_emb(subj_pst)
            obj_pe_inputs = self.pe_emb(obj_pst)
            pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)
            final_hidden = self.attn_layer(outputs, masks, hidden, pe_features)
        else:
            final_hidden = hidden

        return final_hidden

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)

class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.seq_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.label_classifier = FCLayer(
            config.hidden_size * 3,
            config.num_labels,
            0.1,
            use_activation=False,
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.classifier_2 = nn.Linear(config.hidden_size, self.config.num_labels)

        self.softmax = nn.Softmax(dim=0)
        self.init_weights()
        self.output_emebedding = None

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, inputs):
        words, masks = inputs["words"], inputs["masks"]
        pos, ner = inputs["pos"], inputs["ner"]
        # e1_pos = inputs["subj_start"]
        # e2_pos = inputs["obj_start"]
        e1_mask = inputs["e1_mask"]
        e2_mask = inputs["e2_mask"]
        outputs = self.bert(
            words,
            attention_mask=masks,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)


        sequence_output = outputs[0]
        pooled_output = outputs[1]
        # R-BERT's embedding fusion
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        pooled_h = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)
        concat_h = torch.cat([pooled_h, e1_h, e2_h], dim=-1)
        logits = self.classifier(concat_h)
      
        return logits, None
        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # return outputs  # (loss), logits, (hidden_states), (attentions), (self.output_emebedding)