"""
Utilities to process data (pre-processing, build vocab, generate samples and
batches.

Much of this code comes from:
Stanford TensorFlow for Deep Learning Research course
https://github.com/chiphuyen/tf-stanford-tutorials/blob/master/examples/process_data.py
Oxford Deep NLP course
https://github.com/oxford-cs-deepnlp-2017/practical-1/blob/master/practical1.ipynb
"""


from collections import Counter
import random
import os
import zipfile
import re
import urllib.request
import lxml.etree

import numpy as np
from six.moves import urllib


# Parameters for downloading data
DOWNLOAD_URL = 'https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip'
EXPECTED_BYTES = 31344016
DATA_FOLDER = 'data/'
FILE_NAME = 'ted_en-20160408'


def download(file_name):
    """Download the dataset text8 if it's not already downloaded."""
    file_path = DATA_FOLDER + file_name + '.zip'
    if os.path.exists(file_path):
        print('Dataset ready')
    else:
        urllib.request.urlretrieve(DOWNLOAD_URL, file_path)
    return file_path


def read_data(file_name):
    """Read data into text."""
    file_path = DATA_FOLDER + file_name
    with zipfile.ZipFile(file_path + '.zip', 'r') as z:
        doc = lxml.etree.parse(z.open(file_name + '.xml', 'r'))
    raw_text = '\n'.join(doc.xpath('//content/text()'))
    del doc
    return raw_text


def data_preprocessing(raw_text):
    """Clean raw text file."""
    # remove explanations that are enclosed in parenthesis (not part of acutal talk)
    raw_text = re.sub(r'\([^)]*\)', '', raw_text)
    sentences = []
    # remove speakers names
    for line in raw_text.split('\n'):
        m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
        sentences.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)
    sentences_str = ' '.join(sentences)
    # strip unnecessary characters and split into words
    words = re.sub(r'[^a-z0-9]+', ' ', sentences_str.lower()).split()
    return words


def build_vocab(words, vocab_size):
    """Build vocabulary of VOCAB_SIZE most frequent words."""
    dictionary = dict()
    count = [('UNK', -1)]
    count.extend(Counter(words).most_common(vocab_size - 1))
    index = 0
    # write 1000 most common words to file which is used in t-SNE visualization
    if not os.path.exists('./processed/'):
        os.makedirs('./processed/')
    with open('./processed/vocab_1000.tsv', 'w') as f:
        for word, _ in count:
            dictionary[word] = index
            if index < 1000:
                f.write(word + '\n')
            index += 1
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, index_dictionary


def convert_words_to_index(words, dictionary):
    """Replace each word in the dataset with its index in the dictionary."""
    return [dictionary[word] if word in dictionary else 0 for word in words]


def generate_sample(index_words, context_window_size):
    """Form training pairs according to the skip-gram model."""
    for index, center in enumerate(index_words):
        context = random.randint(1, context_window_size)
        # get a random target before the center word
        for target in index_words[max(0, index - context): index]:
            yield center, target
        # get a random target after the center wrod
        for target in index_words[index + 1: index + context + 1]:
            yield center, target


def get_batch(iterator, batch_size):
    """Group a numerical stream into batches and yield them as Numpy arrays."""
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch


def process_data(vocab_size, batch_size, skip_window):
    """Process raw data (clean, build vocab, convert text to ids, generate
    batches.
    """
    _ = download(FILE_NAME)
    raw_text = read_data(FILE_NAME)
    words = data_preprocessing(raw_text)

    # build dictionary and convert words to ids
    dictionary, _ = build_vocab(words, vocab_size)
    index_words = convert_words_to_index(words, dictionary)
    del words  # to save memory

    single_gen = generate_sample(index_words, skip_window)
    return get_batch(single_gen, batch_size)


def get_index_vocab(vocab_size):
    """Get index of words."""
    file_path = download(FILE_NAME, EXPECTED_BYTES)
    words = read_data(file_path)
    return build_vocab(words, vocab_size)


