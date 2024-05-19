import functools
import itertools
import random

import nltk.corpus
from nltk import ngrams, FreqDist
from nltk.corpus import gutenberg
from nltk.probability import FreqDist


def load_corpus():
    return gutenberg.words('shakespeare-macbeth.txt')


def word_frequency(corpus: list[str]) -> FreqDist:
    return FreqDist(corpus)


def distinct_words(corpus: list[str]) -> int:
    return len(word_frequency(corpus))


def ngram_frequency(corpus: object, n: int) -> FreqDist:
    return FreqDist(ngrams(corpus, n))


def bigram_frequency(corpus):
    return ngram_frequency(corpus, 2)


def trigram_frequency(corpus):
    return ngram_frequency(corpus, 3)


class LanguageGenerator:

    def __init__(self, corpus, gram_order, number_of_words, frequency_distribution, conditional_random_sampler):
        self.corpus = corpus
        self.gram_order = gram_order
        self.number_of_words = number_of_words
        self.frequency_distribution = frequency_distribution
        self.conditional_random_sampler = conditional_random_sampler

    def initial_words(self):
        return list(choices_facade(population=list(self.frequency_distribution.keys()),
                                   weights=list(self.frequency_distribution.values())))

    def number_of_iterations(self):
        return self.number_of_words - self.gram_order

    @staticmethod
    def concatenate_strings(strings):
        return ' '.join(strings)

    def __call__(self):
        res = self.initial_words()
        for _ in range(self.number_of_iterations()):
            res.append(self.conditional_random_sampler(res))
        return self.concatenate_strings(res)

    # def append_conditional_random_sample(self, res):
    #     return self.conditional_random_sampler(res)

    # def __call__(self):
    #     return self.concatenate_strings(
    #         itertools.islice(
    #             iterate(
    #                 self.append_conditional_random_sample,
    #                 self.initial_words()),
    #             self.number_of_iterations()))


def choices_facade(population, weights):
    return random.choices(population, weights)[0]


def create_language_generator(corpus, gram_order, number_of_words):
    frequency_distribution = ngram_frequency(corpus, gram_order)
    conditional_random_sampler = ConditionalRandomSampler(frequency_distribution, gram_order)
    return LanguageGenerator(corpus, gram_order, number_of_words, frequency_distribution, conditional_random_sampler)


def generated_language(corpus, gram_order, number_of_words):
    res = initial_words(corpus, gram_order)
    conditional_random_sampler = ConditionalRandomSampler(ngram_frequency(corpus, gram_order), gram_order)
    for _ in range(number_of_iterations(gram_order, number_of_words)):
        res.append(conditional_random_sampler(res))
    return ' '.join(res)


def initial_words(corpus, gram_order):
    frequency = ngram_frequency(corpus, gram_order)
    return list(random.choices(population=list(frequency.keys()), weights=list(frequency.values()), k=1)[0])


def number_of_iterations(gram_order, number_of_words):
    return number_of_words - gram_order


class ConditionalRandomSampler:

    def __init__(self, frequency_distribution, gram_order):
        self.frequency_distribution = frequency_distribution
        self.gram_order = gram_order

    def n_gram_starts_with_n_minus_1_last_words(self, generated_so_far, item):
        n_gram, _ = item
        if self.gram_order == 1:
            flag = True
        else:
            flag = tuple(generated_so_far[-(self.gram_order - 1):]) == n_gram[:self.gram_order - 1]
        return flag

    def __call__(self, generated_so_far):
        return choices_facade(
            *zip(
                *filter(
                    functools.partial(self.n_gram_starts_with_n_minus_1_last_words,generated_so_far),
                    self.frequency_distribution.items())))[-1]


def unfoldr(f, seed):
    s = seed
    while True:
        s, t = f(s)
        yield t


def n_gram_matches_words(words, item):
    n_gram, _ = item
    return n_gram[:-1] == words


def next_words(frequency_distribution, current_words):
    next_word = random.choices(*zip(*filter(functools.partial(n_gram_matches_words, current_words),
                                            frequency_distribution.items())))[0][-1]
    return (current_words + (next_word,))[1:], next_word


def generated_language_functional(corpus, gram_order, n_words):
    frequency_distribution: FreqDist = ngram_frequency(corpus, gram_order)
    initial_words1 = random.choices(*zip(*frequency_distribution.items()))[0][1:]
    return ' '.join((list(initial_words1) +
                     list(itertools.islice(
                             unfoldr(functools.partial(next_words, frequency_distribution),
                                     initial_words1),
                             n_words - gram_order + 1))))


def iterate(f, seed):
    return functools.reduce(lambda x, _: f(x), itertools.count(), seed)


def nest(f, seed, n):
    return itertools.islice(iterate(f, seed), n)


def nest_while(f, seed, predicate):
    return itertools.takewhile(predicate, iterate(f, seed))


def append_function_result(f, x):
    return x + [f(x)]


def nest_list(f, seed, n):
    return nest(functools.partial(append_function_result, f), [seed], n)


def nest_list_while(f, seed, predicate):
    return nest_while(functools.partial(append_function_result, f), [seed], predicate)


if __name__ == '__main__':
    corpus1 = load_corpus()
    # print(generated_language(corpus1, 1, 100))
    # language_generator = create_language_generator(corpus1, 6, 100)
    # print(language_generator())
    print(generated_language_functional(corpus1, 7, 200))
