from collections.abc import Mapping
from unittest import mock

from pytest import fixture

import ex_22_1
from ex_22_1 import bigram_frequency, generated_language, trigram_frequency, distinct_words, load_corpus, \
    word_frequency, initial_words, number_of_iterations


@fixture
def corpus():
    return load_corpus()


def test_load_corpus(corpus):
    assert len(corpus) >= 1e5


def test_word_frequency(corpus):
    assert isinstance(word_frequency(corpus), Mapping)


def test_distinct_words(corpus):
    assert distinct_words(corpus) >= 0


def test_bigram_frequency(corpus):
    assert isinstance(bigram_frequency(corpus), Mapping)


def test_trigram_frequency(corpus):
    assert isinstance(trigram_frequency(corpus), Mapping)


@mock.patch('ex_22_1.ngram_frequency')
@mock.patch('ex_22_1.RandomSample.__call__')
@mock.patch('ex_22_1.number_of_iterations')
@mock.patch('ex_22_1.initial_words')
def test_generated_language(initial_words1, number_of_iterations1, random_sample, ngram_frequency, corpus):
    initial_words1.return_value = ['a']
    number_of_iterations1.return_value = 1
    random_sample.return_value = 'b'

    res = generated_language(corpus, mock.sentinel.gram_order, mock.sentinel.number_of_words)

    assert res == 'a b'
    assert initial_words1.called_with(corpus, mock.sentinel.gram_order)
    assert number_of_iterations1.called_with(mock.sentinel.gram_order, mock.sentinel.number_of_words)
    assert random_sample.called_with(corpus, mock.sentinel.gram_order)


def test_initial_words(corpus):
    assert 2 == len(initial_words(corpus, 2))


def test_number_of_iterations():
    number_of_words = 10
    gram_order = 2
    assert number_of_words - gram_order == number_of_iterations(gram_order, number_of_words)


