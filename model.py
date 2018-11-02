import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init

class NeuralDependencyParser(nn.Module):

    def __init__(self, w_size, w_embed_dim, t_size, t_embed_dim, hidden_size, label_size, layer_num, dropout, feature_num, pretrained_w2v = None, pretrain_t2v = None, scope = 1., multi = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.dropout = dropout
        self.feature_num = feature_num
        self.out = nn.Linear(self.hidden_size * self.feature_num, self.label_size)


        self.multi = multi

        if self.multi == False:
            self.w_embed = nn.Embedding(w_size, w_embed_dim)
            self.t_embed = nn.Embedding(t_size, t_embed_dim)

            self.bilstm = nn.LSTM(input_size=self.w_embed_dim + self.t_embed_dim,
                                  hidden_size=self.hidden_size // 2,
                                  bidirectional=True,
                                  dropout=self.dropout,
                                  num_layers=layer_num)

            if pretrained_w2v is not None:
                self.w_embed.weight = nn.Parameter(pretrained_w2v)
            else:
                self.w_embed.weight.data.uniform_(-scope, scope)

            if pretrain_t2v is not None:
                self.t_embed.weight = nn.Parameter(pretrain_t2v)
            else:
                self.t_embed.weight.data.uniform_(-scope, scope)


    def forward(self, features, *args):
        #
        # tokens: word2index Tensor, B*Sent_len
        # tags: tag2index Tensor, B*Sent_len
        # features: feature2index Tensor, B*Predict_len*Feature_num
        if self.multi == True:
            encode = args[0]
        else:
            tokens = args[0]
            tags = args[1]
            embed = torch.cat((self.w_embed(tokens),self.t_embed(tags)), dim = -1)
            # B * Sent_len * D
            encode, _ = self.bilstm(embed)
        features = torch.gather(encode, dim= -1, index = features)
        predict = self.out(features)
        return predict




class CRF(nn.Module):
    # crf module for tagging
    # rewrote the tutorial of pytorch, changed into a mini-batch form
    # the tutorial is attached to this file in "advanced_tutorial"
    def __init__(self, label_size, is_cuda, start = -1, stop = -2):
        super().__init__()
        self.label_size = label_size + 2
        self.start = start
        self.stop = stop
        self.transitions = nn.Parameter(
            torch.randn(label_size, label_size))
        self._init_weight()
        self.torch = torch.cuda if is_cuda else torch


    def _init_weight(self):
        init.xavier_uniform(self.transitions)
        self.transitions.data[self.start, :].fill_(-10000.)
        self.transitions.data[:, self.stop].fill_(-10000.)

    def _score_sentence(self, input, tags):
        bsz, sent_len, l_size = input.size()
        score = Variable(self.torch.FloatTensor(bsz).fill_(0.))
        s_score = Variable(self.torch.LongTensor([[self.start]]*bsz))

        tags = torch.cat([s_score, tags], dim=-1)
        input_t = input.transpose(0, 1)

        for i, words in enumerate(input_t):
            temp = self.transitions.index_select(1, tags[:, i])
            bsz_t = gather_index(temp.transpose(0, 1), tags[:, i + 1])
            w_step_score = gather_index(words, tags[:, i+1])
            score = score + bsz_t + w_step_score

        temp = self.transitions.index_select(1, tags[:, -1])
        bsz_t = gather_index(temp.transpose(0, 1),
                    Variable(self.torch.LongTensor([self.stop]*bsz)))
        return score+bsz_t

    def forward(self, input):
        bsz, sent_len, l_size = input.size()
        init_alphas = self.torch.FloatTensor(bsz, self.label_size).fill_(-10000.)
        init_alphas[:, self.start].fill_(0.)
        forward_var = Variable(init_alphas)

        input_t = input.transpose(0, 1)
        for words in input_t:
            alphas_t = []
            for next_tag in range(self.label_size):
                emit_score = words[:, next_tag].contiguous()
                emit_score = emit_score.unsqueeze(1).expand_as(words)

                trans_score = self.transitions[next_tag, :].view(1, -1).expand_as(words)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var, True))
            forward_var = torch.cat(alphas_t, dim=-1)

        return log_sum_exp(forward_var)

    def viterbi_decode(self, input):
        backpointers = []
        bsz, sent_len, l_size = input.size()

        init_vvars = self.torch.FloatTensor(bsz, self.label_size).fill_(-10000.)
        init_vvars[:, self.start].fill_(0.)
        forward_var = Variable(init_vvars)

        input_t = input.transpose(0, 1)
        for words in input_t:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.label_size):
                _trans = self.transitions[next_tag].view(1, -1).expand_as(words)
                next_tag_var = forward_var + _trans
                best_tag_scores, best_tag_ids = torch.max(next_tag_var, 1, keepdim=True) # bsz
                bptrs_t.append(best_tag_ids)
                viterbivars_t.append(best_tag_scores)

            forward_var = torch.cat(viterbivars_t, -1) + words
            backpointers.append(torch.cat(bptrs_t, dim=-1))

        terminal_var = forward_var + self.transitions[self.stop].view(1, -1).expand(bsz, l_size)
        _, best_tag_ids = torch.max(terminal_var, 1)

        best_path = [best_tag_ids.view(-1,1)]
        for bptrs_t in reversed(backpointers):
            best_tag_ids = gather_index(bptrs_t, best_tag_ids)
            best_path.append(best_tag_ids.contiguous().view(-1, 1))

        best_path.pop()
        best_path.reverse()

        return torch.cat(best_path, dim=-1)




def log_sum_exp(input, keepdim=False):
    assert input.dim() == 2
    max_scores, _ = input.max(dim=-1, keepdim=True)
    output = input - max_scores.expand_as(input)
    return max_scores + torch.log(torch.sum(torch.exp(output), dim=-1, keepdim=keepdim))

def gather_index(input, index):
    assert input.dim() == 2 and index.dim() == 1
    index = index.unsqueeze(1).expand_as(input)
    output = torch.gather(input, 1, index)
    return output[:, 0]


class MultiLSTMEncode(nn.Module):
    def __init__(self, w_size, w_embed_dim, t_size, t_embed_dim, hidden_size, layer_num, dropout, pretrained_w2v = None, pretrain_t2v = None, scope = 1.):
        super().__init__()
        self.w_embed = nn.Embedding(w_size, w_embed_dim)
        self.w_embed_dim = w_embed_dim
        self.t_embed = nn.Embedding(t_size, t_embed_dim)
        self.t_embed_dim = t_embed_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.bilstm = nn.LSTM(input_size= self.w_embed_dim+self.t_embed_dim,
                              hidden_size=self.hidden_size // 2,
                              bidirectional=True,
                              dropout= self.dropout,
                              num_layers = layer_num,
                              batch_first= True
                              )
        if pretrained_w2v is not None:
            self.w_embed.weight = nn.Parameter(pretrained_w2v)
        else:
            self.w_embed.weight.data.uniform_(-scope, scope)

        if pretrain_t2v is not None:
            self.t_embed.weight = nn.Parameter(pretrain_t2v)
        else:
            self.t_embed.weight.data.uniform_(-scope,scope)

    def forward(self, words, tags):
        embed = torch.cat((self.w_embed(words), self.t_embed(tags)), dim = -1)
        encode, _ = self.bilstm(embed)
        return encode


class SRLDecode(nn.Module):
    def __init__(self, w_size, w_embed_dim, input_size, label_size, hidden_size, layer_num, dropout, crf_flag = False):
        self.w_embed = nn.Embedding(w_size, w_embed_dim)
        self.input_size = input_size
        self.label_size = label_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.dropout = dropout
        self.vv_encoder = nn.Linear(w_embed_dim, w_embed_dim)
        self.bilstm = nn.LSTM(input_size= self.w_embed_dim+self.input_size,
                              hidden_size=self.hidden_size // 2,
                              bidirectional=True,
                              dropout= self.dropout,
                              num_layers = layer_num,
                              batch_first=True)
        self.crf_flag = crf_flag
        if self.crf_flag == True:
            self.logistic = nn.Linear(hidden_size, label_size)
            self.crf = CRF(label_size, is_cuda = True)
        if self.crf_flag == False:
            self.decode = nn.Linear(self.hidden_size, self.label_size)


    def forward(self, *args):
        inputs = args[0]
        verbs = args[1]

        if self.crf_flag == True:
            labels = args[2]
        v_embed = self.w_embed(verbs)
        v_encode = self.vv_encoder(v_embed)
        sentences = torch.chunk(inputs, inputs.size()[0], dim = 0)
        verbs = torch.chunk(v_encode, v_encode.size()[0], dim = 0)
        tmp = []
        for i, verb in enumerate(verbs):
            # v = tuple([verb]*sentences[i].size()[0])
            # v = torch.cat(v, dim = 0)
            v = verb.repeat(sentences[i].size()[1], 1)
            embed = torch.cat((sentences[i],v.unsqueeze(0)), dim = -1)
            tmp.append(embed)

        embed = torch.cat(tmp, dim = 0)

        out, _ = self.bilstm(embed)
        if self.crf_flag == False:
            out = self.decode(out)
            out = F.log_softmax(out, dim = -1)
        else:
            output = self.logistic(out)
            pred = self.crf(output)
            gold = self.crf._score_sentence(output,labels)
            out = (pred-gold).mean()

        return out

class LSTMSRL(nn.Module):
    def __init__(self, w_size, w_embed_dim, t_size, t_embed_dim,
                 encoder_hidden_size, encoder_layer_num, label_size,
                 decoder_hidden_size, decoder_layer_num, dropout,
                 pretrained_w2v = None, pretrain_t2v = None, scope = 1., crf_flag = False):
        super().__init__()
        self.crf_flag = crf_flag
        self.encoder = MultiLSTMEncode(w_size, w_embed_dim, t_size, t_embed_dim, encoder_hidden_size, encoder_layer_num, dropout, pretrained_w2v = pretrained_w2v, pretrain_t2v = pretrain_t2v, scope = scope)
        self.decoder = SRLDecode(w_size, w_embed_dim, encoder_hidden_size, label_size, decoder_hidden_size, decoder_layer_num, dropout, crf_flag = crf_flag)

    def forward(self, *args):
        words = args[0]
        tags = args[1]
        verbs = args[2]
        verb_index = args[3]
        if self.crf_flag == True:
            labels = args[4]
        encode = self.encoder(words, tags)
        encode = torch.index_select(encode, dim = 0, index = verb_index)
        if self.crf_flag == False:
            out = self.decoder(encode, verbs)
        else:
            out = self.decoder(encode, verbs, labels, crf_flag=True)
        return out


class MultiLSTMSRL(nn.Module):
    def __init__(self, w_size, w_embed_dim, t_size, t_embed_dim,
                 encoder_hidden_size, encoder_layer_num, label_size,
                 decoder_hidden_size, decoder_layer_num, dropout,
                 pretrained_w2v = None, pretrain_t2v = None, scope = 1., crf_flag = False):
        super().__init__()
        self.crf_flag = crf_flag
        self.encoder = MultiLSTMEncode(w_size, w_embed_dim, t_size, t_embed_dim, encoder_hidden_size, encoder_layer_num, dropout, pretrained_w2v = pretrained_w2v, pretrain_t2v = pretrain_t2v, scope = scope)
        self.decoder = SRLDecode(w_size, w_embed_dim, encoder_hidden_size, label_size, decoder_hidden_size, decoder_layer_num, dropout, crf_flag = crf_flag)
        self.parser = NeuralDependencyParser(w_size, w_embed_dim, t_size, t_embed_dim, decoder_hidden_size, label_size, decoder_layer_num, dropout, feature_num = 4,pretrained_w2v = pretrained_w2v, pretrain_t2v = pretrain_t2v, scope = scope)

    def forward(self, features, *args):
        words = args[0]
        tags = args[1]
        verbs = args[2]
        verb_index = args[3]
        if self.crf_flag == True:
            labels = args[4]
        encode = self.encoder(words, tags)
        encode = torch.index_select(encode, dim = 0, index = verb_index)
        if self.crf_flag == False:
            out = self.decoder(encode, verbs)
        else:
            out = self.decoder(encode, verbs, labels, crf_flag=True)
        parser_out = self.parser(features, encode)
        return out, parser_out

class SALSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional, dropout, num_layers, label_num):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_layers = num_layers
        self.label_num = label_num
        self.label_embedding = nn.Embedding(label_num, hidden_size)

    def forward(self):
        return


class MultiSelfAttentionEncode(nn.Module):
    def __init__(self, w_size, w_embed_dim, t_size, t_embed_dim, hidden_size, layer_num, dropout, mh_num,
                 pretrained_w2v=None, pretrain_t2v=None, scope=1.):
        super().__init__()
        self.w_embed = nn.Embedding(w_size, w_embed_dim)
        self.w_embed_dim = w_embed_dim
        self.t_embed = nn.Embedding(t_size, t_embed_dim)
        self.t_embed_dim = t_embed_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.mh_num = mh_num

        self.bilstm = nn.LSTM(input_size=self.w_embed_dim + self.t_embed_dim,
                              hidden_size=self.hidden_size,
                              bidirectional=True,
                              dropout=self.dropout,
                              num_layers=layer_num)
        if pretrained_w2v is not None:
            self.w_embed.weight = nn.Parameter(pretrained_w2v)
        else:
            self.w_embed.weight.data.uniform_(-scope, scope)

        if pretrain_t2v is not None:
            self.t_embed.weight = nn.Parameter(pretrain_t2v)
        else:
            self.t_embed.weight.data.uniform_(-scope, scope)



