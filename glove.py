class GloVe:
    def __init__(self, path, dim = 100):
        self.path = path
        self.dim = dim
        self.__load_embedding()
    def __load_embedding(self):
        self.embedding = []
        self.word2index = {}
        self.index2word = {}

        for i, line in enumerate(open(self.path)):
            line = line.strip().split()
            if len(line) <= self.dim:
                continue
            word = line[0]
            vector = list(map(float, line[-self.dim:]))
            assert(len(vector)==self.dim)
            self.embedding.append(vector)
            self.word2index[word] = i
            self.index2word[i] = word


    def get_matrix(self):
        return self.embedding

    def get_word_size(self):
        return len(self.embedding)

    def get_word2index(self):
        return self.word2index

    def get_index2word(self):
        return self.index2word

    def get_dim(self):
        return self.dim


    def add_token(self, word, init = None):
        if init:
            self.embedding.append(init)

        else:
            self.embedding.append([0]*self.dim)

        self.index2word[len(self.index2word)] = word
        self.word2index[word] = len(self.word2index)





