import os


class Sentence():
    def __init__(self, words, tags):
        self.words = words
        self.tags = tags

def load_pos(path):
    data = []
    pathlist = os.listdir(path)
    for p in pathlist:
        p = os.path.join(path,p)
        if os.path.isfile(p):
            sentences = []
            f = open(p)
            lines = f.readlines()
            for line in lines:
                if line[0] != '<' and line[0]!=' ':
                    words = []
                    tags = []
                    tokens = line.split()
                    for token in tokens:
                        tmp = token.split('_')
                        words.append(tmp[0])
                        tags.append(tmp[1])
                    s = Sentence(words,tags)
                    sentences.append(s)
            data.append(sentences)
    return data


def bracket_combine(path):
    data = []
    pathlist = os.listdir(path)
    for p in pathlist:
        if p[0] != '.':
            p = os.path.join(path,p)
            if os.path.isfile(p):
                f = open(p,'r')
                lines = f.readlines()
                for line in lines:
                    if line[0]!='<' and '<DATE>' not in line:
                        data.append(line)
                    elif line =='</S>\n':
                        data.append('\n')
    return data

def write_bracket_full(path):
    data = bracket_combine(path)
    f = open('/Users/yingliu/PycharmProjects/MultiSRL/data/ctb','w')
    for d in data:
        print(d,file = f,end = '')

def convert_dependency():
    os.system('java -jar /Users/yingliu/Downloads/Penn2Malt.jar /Users/yingliu/PycharmProjects/MultiSRL/data/ctb /Users/yingliu/Downloads/chn_headrules.txt 3 2 chtb')


# path = '/Users/yingliu/Downloads/ctb9.0/data/bracketed/'
# write_bracket_full(path)
# convert_dependency()


def load_seg(path):
    data = []
    pathlist = os.listdir(path)
    for p in pathlist:
        p = os.path.join(path, p)
        if os.path.isfile(p):
            f = open(p)
            lines = f.readlines()
            for line in lines:
                if line[0] != '<' and line[0] != ' ':
                    tokens = line.split()
                    data.append(tokens)
    return data

def write_seq_full(path):
    data = load_seg(path)
    f = open('/Users/yingliu/PycharmProjects/MultiSRL/data/ctb_seg.txt','w')
    for d in data:
        for token in d:
            print(token, file=f, end=' ')
        print('',file = f)




# path = '/Users/yingliu/Downloads/ctb9.0/data/segmented/'
# write_seq_full(path)

def write_dep_sorted():
    f_seg = open('/Users/yingliu/PycharmProjects/MultiSRL/data/ctb_seg.txt','r')
    f_dep = open('/Users/yingliu/PycharmProjects/MultiSRL/data/ctb.3.pa.gs.tab','rb')
    f_full = open('/Users/yingliu/PycharmProjects/MultiSRL/data/ctb_dep.txt','w')
    lines = f_seg.readlines()
    for line in lines:
        tokens = line.split()
        for token in tokens:
            info = f_dep.readline()
            t = info.split()
            try:
                if t[0].decode('utf-8') == '<':
                    print('Another BUG hahaha')
                    while True:
                        info = f_dep.readline()
                        t = info.split()
                        if t[0].decode('utf-8') == '>' or t[0].decode('utf-8')=='"':
                            f_dep.readline()
                            info = f_dep.readline()
                            t = info.split()
                            break
            except:
                print('bug')


            byts = t[1:]
            dep_str = []
            for by in byts:
                dep_str.append(by.decode('utf-8'))
            tmp = [token] + dep_str
            for t in tmp:
                print(t,file= f_full, end = ' ')
            print('',file = f_full)
        f_dep.readline()
        print('', file=f_full)

from collections import deque

def transform_transition(dep, type):
    stack = [[0, -1, 'R']]
    transitions = []
    buffer = deque()
    buffer_dep = deque()
    for i in range(len(dep)):
        buffer.append([i+1 , dep[i], type[i]])
        buffer_dep.append(dep[i])
    transitions.append('SH')
    stack.append(buffer[0])
    buffer.popleft()
    buffer_dep.popleft()
    while True:
        if len(stack) < 2:
            if len(buffer) == 0:
                break
            else:
                stack.append(buffer[0])
                buffer.popleft()
                buffer_dep.popleft()
                transitions.append('SH')
        # for b in buffer:
        #     if b[1]== stack[-1][0]:
        #         stack.append(buffer[0])
        #         buffer.popleft()
        #         transitions.append('SH')
        elif (stack[-2][1] == stack[-1][0] and stack[-2][0] not in buffer_dep):
            transitions.append('LEFT-ARC-'+stack[-2][2])
            stack.pop(-2)
        elif (stack[-1][1] == stack[-2][0] and stack[-1][0] not in buffer_dep):
            transitions.append('RIGHT-ARC-'+stack[-1][2])
            stack.pop()
        else:
            try:
                stack.append(buffer[0])
                buffer.popleft()
                buffer_dep.popleft()
                transitions.append('SH')
            except:
                print(buffer)
                print(stack)
                print(transitions)
                print(buffer_dep)
                print(dep)

    return transitions

def transition_full():
    f_full = open('/Users/yingliu/PycharmProjects/MultiSRL/data/ctb_dep.txt', 'r')
    f_transition = open('/Users/yingliu/PycharmProjects/MultiSRL/data/ctb_transition.txt', 'w')
    sentence = []
    for line in f_full.readlines():
        if len(line.split()) == 0:
            dep = []
            type = []
            words = []
            tags = []
            for t in sentence:
                try:
                    words.append(t[0])
                    tags.append(t[1])
                    dep.append(int(t[2]))
                    type.append(t[3])
                except IndexError:
                    print(t)
                    print(sentence)

            try:
                transition = transform_transition(dep,type)
            except:
                print(sentence)
            for w in words:
                print(w, file = f_transition, end = ' ')
            print('',file=f_transition)
            for t in tags:
                print(t, file=f_transition, end=' ')
            print('', file=f_transition)
            for trans in transition:
                print(trans, file=f_transition, end=' ')
            print('', file=f_transition)
            print('', file=f_transition)
            sentence = []
        else:
            tokens = line.split()
            sentence.append(tokens)


def load_transition_treebank(path):
    f = open(path,'r')
    words = []
    tags = []
    transitions = []
    flag = 0
    for line in f.readlines():
        tmp = line.split()
        if len(tmp) == 0:
            flag = 0
            continue
        elif flag == 0:
            words.append(tmp)
        elif flag == 1:
            tags.append(tmp)
        elif flag == 2:
            transitions.append(tmp)

        flag += 1

    return words, tags, transitions




# write_dep_sorted()

# transition_full()
# dep=[2,6,6,6,6,7,0]
# type = ['NMOD','NMOD','NMOD','NMOD','NMOD','SUB','ROOT']
# print(transform_transition(dep,type))

# d = deque([1,2,3,4,5])
# d.pop()
# d.popleft()
# print(len(d))





