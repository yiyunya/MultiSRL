from load_treebank import Sentence
from collections import OrderedDict

def load_text(path):
    f = open(path, 'r')
    lines = f.readlines()
    sentences = []
    words = []
    tags = []
    for line in lines:
        tmp = line.split()
        if len(tmp) != 0:
            words.append(tmp[0])
            tags.append(tmp[1])
        else:
            sentences.append(Sentence(words,tags))
            words = []
            tags = []

    return sentences


class PropSentence():
    def __init__(self, v_num, v_text, v_position, labels):
        self.v_num = v_num
        self.v_text = v_text
        self.v_position = v_position
        self.labels = labels
        self.words = None
        self.tags = None

    def add_words_tags(self, words, tags):
        self.words = words
        self.tags = tags


    def print_sentence(self):
        print('Sentence State:')
        print('v_num: %d'% (self.v_num))
        print('v_text: ' + str(self.v_text))
        print('words: ' + str(self.words))







def load_prop_state(path):
    f = open(path, 'r')
    lines = f.readlines()
    sentences = []
    v_num = 0
    v_text = []
    v_position = []
    labels = {}
    helper = []

    flag = 0
    for line_num,line in enumerate(lines):
        tmp = line.split()
        if len(tmp) == 0:
            for i, text in enumerate(v_text):
                labels[text] = helper[i]
            sentences.append(PropSentence(v_num,v_text,v_position,labels))
            labels = {}
            helper = []
            flag = 0
            continue
        elif flag == 0:
            v_num = len(tmp) - 1
            v_text = [0]*v_num
            v_position = [0]*v_num
            for i in range(v_num):
                helper.append([tmp[i + 1]])
        else:
            for i in range(v_num):
                helper[i].append(tmp[i + 1])
        if tmp[0]!= '-':
            position = flag
            text = tmp[0]
            index = tmp.index('(V*V)')
            v_text[index - 1] = text
            v_position[index - 1] = position
        flag += 1


    return sentences

def load_props():
    props_path = '/Users/yingliu/PycharmProjects/MultiSRL/data/trn/trn.props'
    text_path = '/Users/yingliu/PycharmProjects/MultiSRL/data/trn/trn.text'
    w_s = load_text(text_path)
    p_s = load_prop_state(props_path)
    sentences = []
    for w, p in zip(w_s, p_s):
        p.add_words_tags(w.words, w.tags)
        sentences.append(p)
    # sentences = load_prop_state(props_path)
    return sentences



sentences = load_props()
sentences[0].print_sentence()









