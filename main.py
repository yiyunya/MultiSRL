import argparse
from train import LSTMSRLModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", type=str, dest="train_path",
                        default='./data/train/')
    parser.add_argument("-d", "--dev", type=str, dest="dev_path",
                        default='./data/dev/')
    parser.add_argument('--epochs', type=int, default=32,
                        help='number of epochs for train')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size for training')
    parser.add_argument('--cuda-able', default=True,
                        help='enables cuda')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--use-crf', type = bool, default=False,
                        help='use crf')
    parser.add_argument('--save', type=str, default='./lstm.pt',
                        help='path to save the final model')
    parser.add_argument('--save-epoch', action='store_true',
                        help='save every epoch')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='the probability for dropout')
    parser.add_argument('--encode-lstm-hsz', type=int, default=200,
                        help='Encode BiLSTM hidden size')
    parser.add_argument('--decode-lstm-hsz', type=int, default=300,
                        help='Decode BiLSTM hidden size')
    parser.add_argument('--encode-lstm-layers', type=int, default=1,
                        help='Encode BiLSTM layer numbers')
    parser.add_argument('--decode-lstm-layers', type=int, default=1,
                        help='Decode BiLSTM layer numbers')
    parser.add_argument('--l2', type=float, default=0.05,
                        help='l2 regularization')
    parser.add_argument('--clip', type=float, default=.5,
                        help='gradient clipping')
    args = parser.parse_args()
    print('Options: %s' % args)

    model = LSTMSRLModel(args)
    model.fit()


