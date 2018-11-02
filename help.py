# import random
# hahah = []
# for i in range(20):
#     hahah.append(i+21)
# random.shuffle(hahah)
# print(hahah)

# a = '123'
# print(len(a))


import torch

a = [[1,2,3],[2,3,4]]

b = torch.LongTensor(a)

print(b)

print(b.tolist())


# a = tuple([0]*10)
#
# print(a)


# labels = ['B-A0', 'I-A0', 'E-A0', 'VV', 'U-A1', 'U-AM', 'O', 'B-A1', 'I-A1', 'E-A1', 'B-AM', 'E-AM',
#                'U-A0', 'B-A2', 'I-A2', 'E-A2', 'I-AM', 'U-A2', 'B-A3', 'I-A3', 'E-A3', 'B-A4', 'I-A4', 'E-A4',
#                'U-A4', 'U-A3', '<pad>']
#
# print(len(labels))
# print(labels.index('<pad>'))