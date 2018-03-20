import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens


def get_input_layer(word_idx, vocab_n):
    x = torch.zeros(vocab_n).float()
    x[word_idx] = 1.0
    return x


def main():
    corpus = [
            'he is a king',
            'she is a queen',
            'he is a man',
            'she is a woman',
            'warsaw is poland capital',
            'berlin is germany capital',
            'paris is france capital',
            ]

    tokenized_corpus = tokenize_corpus(corpus)

    vocab = []
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocab:
                vocab.append(token)
    w2i = {w: idx for (idx, w) in enumerate(vocab)}
    i2w = {idx: w for (idx, w) in enumerate(vocab)}

    vocab_n = len(vocab)

    window_size = 2
    idx_pairs = []
    for sent in tokenized_corpus:
        indices = [w2i[word] for word in sent]
        for center_word_pos in range(len(indices)):
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                if (
                        context_word_pos < 0 or
                        context_word_pos >= len(indices) or
                        center_word_pos == context_word_pos
                   ):
                    continue
                context_word_idx = indices[context_word_pos]
                idx_pairs.append((indices[center_word_pos], context_word_idx))

    idx_pairs = np.array(idx_pairs)

    embedding_n = 5
    W1 = Variable(torch.randn(embedding_n,
                              vocab_n).float(),
                  requires_grad=True)
    W2 = Variable(torch.randn(vocab_n,
                              embedding_n).float(),
                  requires_grad=True)

    epoch = 201
    lr = 0.001

    for i_epoch in range(epoch):
        loss_val = 0
        for data, target in idx_pairs:
            x = Variable(get_input_layer(data, vocab_n)).float()
            y_true = Variable(torch.from_numpy(np.array([target])).long())

            z1 = torch.matmul(W1, x)
            z2 = torch.matmul(W2, z1)

            log_softmax = F.log_softmax(z2, dim=0)

            loss = F.nll_loss(log_softmax.view(1, -1), y_true)
            loss_val += loss.data[0]
            loss.backward()
            W1.data -= lr * W1.grad.data
            W2.data -= lr * W2.grad.data

            W1.grad.data.zero_()
            W2.grad.data.zero_()

        if i_epoch % 10 == 0:
            print(f'Loss at epo {i_epoch}: {loss_val/len(idx_pairs)}')

    return W1, w2i, i2w
