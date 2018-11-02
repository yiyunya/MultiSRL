import torch
from glove import GloVe
from model import LSTMSRL, MultiLSTMSRL
import time
import numpy as np
import load_props as t
import load_dev as d

from load_treebank import load_transition_treebank

from torch.autograd import Variable


class TransitionState(object):

    def __init__(self, tagged_sent):
        self.root = ('ROOT', '<root>', -1)
        self.stack = [self.root]
        self.buffer = [(s[0], s[1], i) for i, s in enumerate(tagged_sent)]
        self.address = [s[0] for s in tagged_sent] + [self.root[0]]
        self.arcs = []
        self.terminal = False

    def __str__(self):
        return 'stack : %s \nbuffer : %s' % (str([s[0] for s in self.stack]), str([b[0] for b in self.buffer]))

    def shift(self):

        if len(self.buffer) >= 1:
            self.stack.append(self.buffer.pop(0))
        else:
            print("Empty buffer")

    def left_arc(self, relation=None):

        if len(self.stack) >= 2:
            arc = {}
            s2 = self.stack[-2]
            s1 = self.stack[-1]
            arc['graph_id'] = len(self.arcs)
            arc['form'] = s1[0]
            arc['addr'] = s1[2]
            arc['head'] = s2[2]
            arc['pos'] = s1[1]
            if relation:
                arc['relation'] = relation
            self.arcs.append(arc)
            self.stack.pop(-2)

        elif self.stack == [self.root]:
            print("Element Lacking")

    def right_arc(self, relation=None):

        if len(self.stack) >= 2:
            arc = {}
            s2 = self.stack[-2]
            s1 = self.stack[-1]
            arc['graph_id'] = len(self.arcs)
            arc['form'] = s2[0]
            arc['addr'] = s2[2]
            arc['head'] = s1[2]
            arc['pos'] = s2[1]
            if relation:
                arc['relation'] = relation
            self.arcs.append(arc)
            self.stack.pop(-1)

        elif self.stack == [self.root]:
            print("Element Lacking")

    def get_left_most(self, index):
        left = ['<NULL>', '<NULL>', None]

        if index == None:
            return left
        for arc in self.arcs:
            if arc['head'] == index:
                left = [arc['form'], arc['pos'], arc['addr']]
                break
        return left

    def get_right_most(self, index):
        right = ['<NULL>', '<NULL>', None]

        if index == None:
            return right
        for arc in reversed(self.arcs):
            if arc['head'] == index:
                right = [arc['form'], arc['pos'], arc['addr']]
                break
        return right

    def is_done(self):
        return len(self.buffer) == 0 and self.stack == [self.root]

    def to_tree_string(self):
        if self.is_done() == False:
            return None
        ingredient = []
        for arc in self.arcs:
            ingredient.append([arc['form'], self.address[arc['head']]])
        ingredient = ingredient[-1:] + ingredient[:-1]
        return self._make_tree(ingredient, 0)

    def _make_tree(self, ingredient, i, new=True):

        if new:
            treestr = "("
            treestr += ingredient[i][0]
            treestr += " "
        else:
            treestr = ""
        ingredient[i][0] = "CHECK"

        parents, _ = list(zip(*ingredient))

        if ingredient[i][1] not in parents:
            treestr += ingredient[i][1]
            return treestr

        else:
            treestr += "("
            treestr += ingredient[i][1]
            treestr += " "
            for node_i, node in enumerate(parents):
                if node == ingredient[i][1]:
                    treestr += self._make_tree(ingredient, node_i, False)
                    treestr += " "

            treestr = treestr.strip()
            treestr += ")"
        if new:
            treestr += ")"
        return treestr


def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<unk>"], seq))
    return Variable(torch.LongTensor(idxs))

class LSTMSRLModel():

    def __init__(self, args):
        self.model = None
        self.args = args
        self.word_embed = GloVe('D:/EMNLP/data/zhgiga100/vectors.txt')
        self.tag_embed = GloVe('D:/EMNLP/data/tagvectors.txt', dim=50)
        self.word_embed.add_token('<pad>')
        self.tag_embed.add_token('<pad>')
        self.tag_embed.add_token('<root>')
        self.word_embed.add_token('ROOT')
        self.word2index = self.word_embed.get_word2index()
        self.index2word = self.word_embed.get_index2word()
        self.tag2index = self.tag_embed.get_word2index()
        self.index2tag = self.tag_embed.get_index2word()
        self.w_dim = self.word_embed.get_dim()
        self.t_dim = self.tag_embed.get_dim()
        self.w_embed = torch.FloatTensor(self.word_embed.get_matrix())
        self.t_embed = torch.FloatTensor(self.tag_embed.get_matrix())
        self.word_size = self.word_embed.get_word_size()
        self.tag_size = self.tag_embed.get_word_size()
        self.use_cuda = torch.cuda.is_available() and args.cuda_able
        self.crf_flag = args.use_crf
        self.encode_lstm_hsz = args.encode_lstm_hsz
        self.decode_lstm_hsz = args.decode_lstm_hsz
        self.encode_layer_num = args.encode_lstm_layers
        self.decode_layer_num = args.decode_lstm_layers
        self.dropout = args.dropout
        self.lr = args.lr
        self.epochs = args.epochs
        self.clip = args.clip
        self.__preprocess_labels()
        self.batch_size = args.batch_size
        self.__prepare_train_dev()

    def __preprocess_labels(self):
        self.label2index = {}
        self.index2label = {}
        self.labels = ['B-A0', 'I-A0', 'E-A0', 'VV', 'U-A1', 'U-AM', 'O', 'B-A1', 'I-A1', 'E-A1', 'B-AM', 'E-AM',
                       'U-A0', 'B-A2', 'I-A2', 'E-A2', 'I-AM', 'U-A2', 'B-A3', 'I-A3', 'E-A3', 'B-A4', 'I-A4', 'E-A4',
                       'U-A4', 'U-A3', '<pad>']
        self.label_size = len(self.labels)
        for i, label in enumerate(self.labels):
            self.label2index[label] = i
            self.index2label[i] = label

    def __prepare_train_dev(self):
        self.__getverblist_and_index(t.load_props())
        self.__getlabelmatrix(t.load_props())
        self.__getmatrix(t.load_props())
        self.__getverblist_and_index(d.load_props(), flag='dev')
        self.__getlabelmatrix(d.load_props(), flag='dev')
        self.__getmatrix(d.load_props(), flag='dev')

    def __getverblist_and_index(self, sentences, flag='train'):
        verblist = []
        verb_index = []
        for i in range(len(sentences)):
            verbs = sentences[i].v_text
            for verb in verbs:
                if verb in self.word2index.keys():
                    verblist.append(self.word2index[verb])
                else:
                    verblist.append(self.word2index["<unk>"])
                verb_index.append(i)
        if flag == 'train':
            self.verblist = verblist
            self.verb_index = verb_index

        else:
            self.dev_verblist = verblist
            self.dev_verb_index = verb_index

    def __getlabelmatrix(self, sentences, flag='train'):
        alllabel = []
        for i in range(len(sentences)):
            verbs = sentences[i].v_text
            label = sentences[i].bie
            for verb in verbs:
                alllabel.append(label[verb])
        labellist = [[self.label2index[l] for l in labels] for labels in alllabel]
        if flag == 'train':
            self.labelmatrix = labellist
        else:
            self.dev_labelmatrix = labellist

    def __getmatrix(self, sentences, flag='train'):
        word = [[self.word2index[w] if w in self.word2index else self.word2index['<unk>'] for w in sentence.words] for
                sentence in sentences]
        if flag == 'train':
            self.wordmatrix = word
        else:
            self.dev_wordmatrix = word
        tag = [[self.tag2index[t] if t in self.tag2index else self.tag2index['<unk>'] for t in sentence.tags] for
               sentence in sentences]
        if flag == 'train':
            self.tagmatrix = tag
        else:
            self.dev_tagmatrix = tag


    def get_feat(self, transition_state, word2index, tag2index, label2index=None):
        word_feats = []
        word_feats.append(transition_state.stack[-1][-1]) if len(transition_state.stack) >= 1 else word_feats.append(
            '<unk>')  # s1
        word_feats.append(transition_state.stack[-2][-1]) if len(transition_state.stack) >= 2 else word_feats.append(
            '<unk>')  # s2


        word_feats.append(transition_state.buffer[0][-1]) if len(transition_state.buffer) >= 1 else word_feats.append(
            '<unk>')  # b1
        word_feats.append(transition_state.buffer[1][-1]) if len(transition_state.buffer) >= 2 else word_feats.append(
            '<unk>')  # b2


        return Variable(torch.LongTensor(word_feats)).view(1, -1)

    def preprocess(self):

        words, tags, transitions = load_transition_treebank('/Users/yingliu/PycharmProjects/MultiSRL/data/ctb_transition.txt')

        self.actions = set(transitions)
        self.action2index = {v: i for i, v in enumerate(transitions)}

        self.train_data = []

        for tx, ty in zip(zip(words,tags), transitions):
            state = TransitionState(tx)
            transition = ty
            while len(transition):
                feat = self.get_feat(state, self.word2index, self.tag2index)
                action = transition.pop(0)
                actionTensor = Variable(torch.LongTensor([self.action2index[action]])).view(1, -1)
                self.train_data.append([feat, actionTensor])
                if action == 'SHIFT':
                    state.shift()
                elif 'RIGHT-ARC' in action:
                    state.right_arc()
                elif 'LEFT-ARC' in action:
                    state.left_arc()

    def build_model(self):
        args = self.args
        self.model = MultiLSTMSRL(w_size=self.word_size, w_embed_dim=self.w_dim,
                             t_size=self.tag_size, t_embed_dim=self.t_dim,
                             encoder_hidden_size=self.encode_lstm_hsz, encoder_layer_num=self.encode_layer_num,
                             label_size=self.label_size, decoder_hidden_size=self.decode_lstm_hsz,
                             decoder_layer_num=self.decode_layer_num, dropout=self.dropout,
                             pretrained_w2v=self.w_embed, pretrain_t2v=self.t_embed, scope=1., crf_flag=self.crf_flag)
        if self.use_cuda:
            self.model = self.model.cuda()
        if self.crf_flag == False:
            self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    @staticmethod
    def get_verb_index(verb_index, sindex, eindex, svindex):
        evindex = 0
        i = eindex - 1
        while i > sindex:
            if i in verb_index:
                evindex = len(verb_index) - verb_index[::-1].index(i)
                return evindex
            i = i - 1
        return svindex

    def get_batch(self, words, tags, verbs, verb_index, labels):
        # print(len(labels))
        # print(len(verbs))
        # print('\n')
        sindex = 0
        svindex = 0
        eindex = self.batch_size
        while eindex < len(words):
            # svindex = len(verb_index) - verb_index[::-1].index(sindex) - 1

            word_batch = words[sindex:eindex]
            tag_batch = tags[sindex:eindex]
            evindex = self.get_verb_index(verb_index, sindex, eindex, svindex)
            verb_batch = verbs[svindex:evindex]
            verb_index_batch = list(map(lambda x: x - sindex, verb_index[svindex:evindex]))
            label_batch = labels[svindex:evindex]

            tmp = eindex
            eindex = eindex + self.batch_size
            sindex = tmp
            svindex = evindex
            yield word_batch, tag_batch, verb_batch, verb_index_batch, label_batch

        if eindex >= len(words):
            word_batch = words[sindex:]
            tag_batch = tags[sindex:]
            verb_batch = verbs[svindex:]
            verb_index_batch = list(map(lambda x: x - sindex, verb_index[svindex:]))
            label_batch = labels[svindex:]
            yield word_batch, tag_batch, verb_batch, verb_index_batch, label_batch

    def pad_batch(self, word_batch, tag_batch, verb_batch, verb_index_batch, label_batch):
        word_max_len = max([len(w) for w in word_batch])
        # print(word_max_len)
        # print('\n')
        word_batch_p, tag_batch_p, label_batch_p = [], [], []
        for word, tag in zip(word_batch, tag_batch):
            word_batch_p.append(word + (word_max_len - len(word)) * [self.word2index['<pad>']])
            tag_batch_p.append(tag + (word_max_len - len(tag)) * [self.tag2index['<pad>']])

        for label in label_batch:
            # print(len(label))
            label_batch_p.append(label + (word_max_len - len(label)) * [self.label2index['<pad>']])
        word_batch_p = Variable(torch.LongTensor(word_batch_p).cuda()).cuda()
        tag_batch_p = Variable(torch.LongTensor(tag_batch_p).cuda()).cuda()
        # try:
        # print('\n')
        # for label in label_batch_p:
        # print(len(label))
        label_batch_p = Variable(torch.LongTensor(label_batch_p).cuda()).cuda()
        # except:
        #    for label in label_batch_p:
        #        print(len(label))
        verb_batch_p = Variable(torch.LongTensor(verb_batch).cuda()).cuda()
        verb_index_batch_p = Variable(torch.LongTensor(verb_index_batch).cuda()).cuda()
        # print('xxx' + str(self.word2index['<pad>']))
        return word_batch_p, tag_batch_p, verb_batch_p, verb_index_batch_p, label_batch_p

    def getBatch(self, batch_size, train_data):
        sindex = 0
        eindex = batch_size
        while eindex < len(train_data):
            batch = train_data[sindex: eindex]
            temp = eindex
            eindex = eindex + batch_size
            sindex = temp
            yield batch

        if eindex >= len(train_data):
            batch = train_data[sindex:]
            yield batch

    def train(self, words, tags, verbs, verb_index, labels):
        self.model.train()
        total_loss = 0
        for (words, tags, verbs, verb_index, label), (i, batch) in zip(self.get_batch(words, tags, verbs, verb_index, labels), enumerate(self.getBatch(self.batch_size, self.train_data))):
            words, tags, verbs, verb_index, label = self.pad_batch(words, tags, verbs, verb_index, label)
            inputs, targets = list(zip(*batch))
            if self.crf_flag == False:
                result = self.model(inputs, words, tags, verbs, verb_index)
                # print(result)
                # print(label)
                # print('\n')
                loss = self.criterion(result[1].view(-1, self.label_size), label.view(-1))
                loss += self.criterion(result[0].view(-1, len(self.actions), targets.view(-1)))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.data

        return total_loss[0] / len(verbs)

    def evaluate(self, words, tags, verbs, verb_index, labels):

        self.model.eval()
        corrects = eval_loss = 0
        count = 0

        for words, tags, verbs, verb_index, label in self.get_batch(words, tags, verbs, verb_index, labels):
            words, tags, verbs, verb_index, label = self.pad_batch(words, tags, verbs, verb_index, label)
            count += label.size()[0] * label.size()[1]
            result = self.model(words, tags, verbs, verb_index)
            loss = self.criterion(result.view(-1, self.label_size), label.view(-1))
            _, pred = torch.max(result, 2)
            eval_loss += loss.data[0]
            corrects += (pred.data == label.data).sum()

        return eval_loss / len(verbs), corrects, corrects * 100 / count, count

    def fit(self):
        # fit the model
        self.build_model()

        train_loss = []
        valid_loss = []
        accuracy = []

        best_acc = None
        total_start_time = time.time()

        try:
            print('-' * 90)
            for epoch in range(1, self.args.epochs + 1):
                epoch_start_time = time.time()
                loss = self.train(self.wordmatrix, self.tagmatrix, self.verblist, self.verb_index, self.labelmatrix)
                train_loss.append(loss * 1000.)

                print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch,
                                                                                      time.time() - epoch_start_time,
                                                                                      loss))

                loss, corrects, acc, size = self.evaluate(self.dev_wordmatrix, self.dev_tagmatrix, self.dev_verblist,
                                                          self.dev_verb_index, self.dev_labelmatrix)

                valid_loss.append(loss * 1000.)
                accuracy.append(acc / 100.)

                v_epoch_start_time = time.time()
                print('-' * 90)
                print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | accuracy {:.4f}%({}/{})'.format(epoch,
                                                                                                             time.time() - v_epoch_start_time,
                                                                                                             loss, acc,
                                                                                                             corrects,
                                                                                                             size))
                print('-' * 90)
                if not best_acc or best_acc < corrects:
                    best_acc = corrects
                    torch.save(self.model.state_dict(), 'D:/EMNLP/data/params.pkl')
        except KeyboardInterrupt:
            print("-" * 90)
            print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time) / 60.0))


'''
Model = LSTMSRLModel(1)
Y_0 = Model.labelmatrix

Y = []
for i in Y_0:
    for j in i:
        Y.append(j)
print(len(Y))

wordlist = []
wordmatrix1 = Model.wordmatrix
for i in wordmatrix1:
    for j in i:
        wordlist.append(j)

print(len(wordlist))
'''