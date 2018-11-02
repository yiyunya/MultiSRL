def make_pos_data(data):
    sents = []
    for doc in data:
        for s in doc:
            sents.append(s.tags)
    f = open('/Users/yingliu/PycharmProjects/MultiSRL/data/tagebdtext.txt','w')
    for sent in sents:
        for tag in sent:
            print(tag, end = ' ',file = f)
        print('\n', end = '', file = f)


from load_treebank import *

data = load_pos(path='/Users/yingliu/Downloads/ctb9.0/data/postagged/')
make_pos_data(data)