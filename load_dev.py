# from load_treebank import Sentence
# from collections import OrderedDict

class Sentence():
    def __init__(self, words, tags):
        self.words = words
        self.tags = tags


def load_text(path):
    f = open(path, 'r', encoding="utf-8")
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
            sentences.append(Sentence(words, tags))
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
        self.generate_bie()

    def add_words_tags(self, words, tags):
        self.words = words
        self.tags = tags

    def print_sentence(self):
        print('Sentence State:')
        print('v_num: %d' % (self.v_num))
        print('v_text: ' + str(self.v_text))
        print('words: ' + str(self.words))
        print(self.labels)

    def generate_bie(self):
        # global alltpye
        self.bie = {}
        for vtext in self.v_text:
            tmplist = self.labels[vtext]
            # print(tmplist)
            bielist = []
            has_begin = 0
            beginword = ''
            for tmpword in tmplist:
                if tmpword[0] == '(' and tmpword[-1] == ')' and tmpword != '(V*V)':
                    bielist.append('U-' + tmpword[1:3])
                if tmpword == '(V*V)':
                    bielist.append('VV')
                if tmpword[0] == '(' and tmpword[-1] != ')':
                    has_begin = 1
                    beginword = tmpword[1:3]
                    bielist.append('B-' + beginword)
                if tmpword == '*':
                    if has_begin == 1:
                        bielist.append('I-' + beginword)
                    else:
                        bielist.append('O')
                if tmpword[0] != '(' and tmpword[-1] == ')':
                    has_begin = 0
                    bielist.append('E-' + beginword)
                    beginword = ''
            # print(bielist)
            # for types in bielist:
            #    if types not in alltype:
            #        alltype.append(types)
            self.bie[vtext] = bielist
        # print(self.bie)


def load_prop_state(path):
    f = open(path, 'r', encoding="utf-8")
    lines = f.readlines()
    sentences = []
    v_num = 0
    v_text = []
    v_position = []
    labels = {}
    helper = []
    label_trans = {'(V*)': '(V*V)', '(A0*)': '(A0*A0)', '(A1*)': '(A1*A1)', '(A2*)': '(A2*A2)', '(A3*)': '(A3*A3)',
                   '(A4*)': '(A4*A4)'}

    flag = 0
    for line_num, line in enumerate(lines):
        tmp = line.split()
        if len(tmp) == 0:
            for i, text in enumerate(v_text):
                labels[text] = helper[i]
            sentences.append(PropSentence(v_num, v_text, v_position, labels))
            labels = {}
            helper = []
            flag = 0
            continue
        elif flag == 0:
            v_num = len(tmp) - 1
            v_text = [0] * v_num
            v_position = [0] * v_num
            for i in range(v_num):
                if tmp[i + 1] in label_trans.keys():
                    helper.append([label_trans[tmp[i + 1]]])
                else:
                    helper.append([tmp[i + 1]])
        else:
            for i in range(v_num):
                if tmp[i + 1] in label_trans.keys():
                    helper[i].append(label_trans[tmp[i + 1]])
                else:
                    helper.append([tmp[i + 1]])
        if tmp[0] != '-':
            position = flag
            text = tmp[0]
            print(tmp)
            index = tmp.index('(V*)')
            v_text[index - 1] = text
            v_position[index - 1] = position
        flag += 1

    return sentences


def load_props():
    props_path = '/data/dev/dev.props'
    text_path = '/data/dev/dev.text'
    w_s = load_text(text_path)
    print(len(w_s))
    p_s = load_prop_state(props_path)
    sentences = []
    for w, p in zip(w_s, p_s):
        p.add_words_tags(w.words, w.tags)
        sentences.append(p)
    # sentences = load_prop_state(props_path)
    return sentences
