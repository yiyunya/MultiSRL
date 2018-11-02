import torch
from glove import GloVe
from model import LSTMSRL
import time
import numpy as np
import load_props as t
import load_dev as d

from torch.autograd import Variable


class LSTMSRLModel():

    def __init__(self, args):
        self.model = None
        self.args = args
        self.word_embed = GloVe('D:/EMNLP/data/zhgiga100/vectors.txt')
        self.tag_embed = GloVe('D:/EMNLP/data/tagvectors.txt', dim=50)
        self.word_embed.add_token('<pad>')
        self.tag_embed.add_token('<pad>')
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

    def build_model(self):
        args = self.args
        self.model = LSTMSRL(w_size=self.word_size, w_embed_dim=self.w_dim,
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




    def train(self, words, tags, verbs, verb_index, labels):
        self.model.train()
        total_loss = 0
        count = 0
        for words, tags, verbs, verb_index, label in self.get_batch(words, tags, verbs, verb_index, labels):
            words, tags, verbs, verb_index, label = self.pad_batch(words, tags, verbs, verb_index, label)
            if self.crf_flag == False:
                result = self.model(words, tags, verbs, verb_index)
                # print(result)
                # print(label)
                # print('\n')
                loss = self.criterion(result.view(-1, self.label_size), label.view(-1))
                count += label.size()[0]*label.size()[1]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.data

        return total_loss[0] / count

    def evaluate(self, words, tags, verbs, verb_index, labels):

        self.model.eval()
        corrects = eval_loss = 0
        count = 0

        for words, tags, verbs, verb_index, label in self.get_batch(words, tags, verbs, verb_index, labels):
            words, tags, verbs, verb_index, label = self.pad_batch(words, tags, verbs, verb_index, label)
            # count += label.size()[0] * label.size()[1]
            result = self.model(words, tags, verbs, verb_index)
            loss = self.criterion(result.view(-1, self.label_size), label.view(-1))
            _, pred = torch.max(result, 2)
            eval_loss += loss.data[0]
            # corrects += (pred.data == label.data).sum()
            tmp_count, tmp_corrects = self.caculate(pred, label)
            count += tmp_count
            corrects += tmp_corrects

        return eval_loss / len(verbs), corrects, corrects * 100 / count, count

    def caculate(self, pred, label):
        label = label.data.tolist()
        pred = pred.data.tolist()
        count = 0
        correct = 0

        for l, p in zip(label, pred):
            for gold, guess in zip(l, p):
                if gold != 26:
                    count += 1
                    if guess == gold:
                        correct += 1
        return count, correct



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