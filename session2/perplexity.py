"""
Calculates perplexity of a model on a dev/test set.
"""

from __future__ import print_function

import argparse
import numpy
from six.moves import cPickle as pkl
from six import iteritems
import theano

from nmt import build_model, load_params, init_params, init_tparams, pred_probs, prepare_data
from data_iterator import TextIterator


def main(model, test_src, test_trg, dictionary_src, dictionary_trg):

    # load model model_options
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)

    # load source dictionary and invert
    with open(dictionary_src, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()

    for kk, vv in iteritems(word_dict):
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load target dictionary and invert
    with open(dictionary_trg, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = dict()
    for kk, vv in iteritems(word_dict_trg):
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    # load data
    data_iter = TextIterator(test_src, test_trg, dictionary_src, dictionary_trg,
                             n_words_source=options['n_words_src'], n_words_target=options['n_words'],
                             batch_size=options['valid_batch_size'], maxlen=100000)

    print('Loading model')
    params = init_params(options)
    params = load_params(model, params)
    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, options)
    inps = [x, x_mask, y, y_mask]

    print('Building f_log_probs...', end="")
    f_log_probs = theano.function(inps, cost, profile=False)
    print('Done')

    # calculate the probabilities
    loss, perplexity = pred_probs(f_log_probs, prepare_data, options, data_iter)
    mean_loss = loss.mean()

    print('Loss: %f' % mean_loss)
    print('Perplexity: %f' % perplexity)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('dictionary_src', type=str)
    parser.add_argument('dictionary_trg', type=str)
    parser.add_argument('test_src', type=str)
    parser.add_argument('test_trg', type=str)

    args = parser.parse_args()

    main(args.model, args.test_src, args.test_trg, args.dictionary_src, args.dictionary_trg)
