"""
These files were taken from the https://github.com/jsbaan/DPAC-DialogueGAN repository to reproduce
the results that were claimed in https://arxiv.org/abs/1701.06547 paper.
"""

import nltk
from collections import Counter
import os

from torch import LongTensor
from torch.utils.data.dataset import Dataset

from torch.utils.data.dataloader import DataLoader


class DPDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=64):
        if dataset == None:
            corpus = DPCorpus(vocabulary_limit=5000)
            dataset = corpus.get_train_dataset(2, 5, 20)

        collator = dataset.corpus.get_collator(reply_length=20)

        super().__init__(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True, drop_last=True)


class DPDataset(Dataset):
    def __init__(self, corpus, dialogs, context_size=2, min_reply_length=None, max_reply_length=None):
        self.corpus = corpus

        self.contexts = []
        self.replies = []
        for dialog in dialogs:
            max_start_i = len(dialog) - context_size
            for start_i in range(max_start_i):
                reply = dialog[start_i + context_size]
                context = []
                for i in range(start_i, start_i + context_size):
                    context.extend(dialog[i])

                if (min_reply_length is None or len(reply) >= min_reply_length) and \
                        (max_reply_length is None or len(reply) <= max_reply_length):
                    self.contexts.append(context)
                    self.replies.append(reply)

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, item):
        context = self.contexts[item]
        replies = self.replies[item]

        return (LongTensor(context), LongTensor(replies))


class DailyDialogParser:
    def __init__(self, path, sos, eos, eou):
        self.path = path
        self.sos = sos
        self.eos = eos
        self.eou = eou

    def get_dialogs(self):
        train_dialogs = self.process_file(self.path + 'train.txt')
        validation_dialogs = self.process_file(self.path + 'validation.txt')
        test_dialogs = self.process_file(self.path + 'test.txt')
        return train_dialogs, validation_dialogs, test_dialogs

    def process_file(self, path):
        with open(path, 'r') as f:
            data = f.readlines()

        print("Parsing", path)
        return [self.process_raw_dialog(line) for line in data]

    def process_raw_dialog(self, raw_dialog):
        raw_utterances = raw_dialog.split('__eou__')
        return [self.process_raw_utterance(raw_utterance) for raw_utterance in raw_utterances if
                not raw_utterance.isspace()]

    def process_raw_utterance(self, raw_utterance):
        raw_sentences = nltk.sent_tokenize(raw_utterance)

        utterence = []
        for raw_sentence in raw_sentences:
            utterence.extend(self.process_raw_sentence(raw_sentence))

        return utterence + [self.eou]

    def process_raw_sentence(self, raw_sentence):
        raw_sentence = raw_sentence.lower()
        raw_sentence = raw_sentence.split()
        return [self.sos] + raw_sentence + [self.eos]


class DPCollator:
    def __init__(self, pad_token, reply_length=None):
        self.pad_token = pad_token
        self.reply_length = reply_length

    def __call__(self, batch):
        contexts, replies = zip(*batch)

        padded_contexts = self.pad(contexts)
        padded_replies = self.pad(replies, self.reply_length)

        return padded_contexts, padded_replies

    def pad(self, data, length=None):
        max_length = length
        if max_length is None:
            max_length = max([len(row) for row in data])

        padded_data = []
        for row in data:
            padding = [self.pad_token] * (max_length - len(row))
            padded_data.append(list(row) + padding)

        return LongTensor(padded_data)


class DPCorpus(object):
    SOS = '<s>'  # Start of sentence token
    EOS = '</s>'  # End of sentence token
    EOU = '</u>'  # End of utterance token
    PAD = '<pad>'  # Padding token
    UNK = '<unk>'  # Unknown token (Out of vocabulary)

    def __init__(self, dialog_parser=None, vocabulary_limit=None):
        if dialog_parser is None:
            path = os.path.dirname(os.path.realpath(__file__)) + '/daily_dialog/'
            dialog_parser = DailyDialogParser(path, self.SOS, self.EOS, self.EOU)

        self.train_dialogs, self.validation_dialogs, self.test_dialogs = dialog_parser.get_dialogs()

        print('Building vocabulary')
        self.build_vocab(vocabulary_limit)

        if vocabulary_limit is not None:
            print('Replacing out of vocabulary from train dialogs by unk token.')
            self.limit_dialogs_to_vocabulary(self.train_dialogs)
            print('Replacing out of vocabulary from validation dialogs by unk token.')
            self.limit_dialogs_to_vocabulary(self.validation_dialogs)
            print('Replacing out of vocabulary from test dialogs by unk token.')
            self.limit_dialogs_to_vocabulary(self.test_dialogs)

    def build_vocab(self, vocabulary_limit):
        special_tokens = [self.PAD, self.UNK]
        all_words = self.flatten_dialogs(self.train_dialogs)

        vocabulary_counter = Counter(all_words)
        if vocabulary_limit is not None:
            vocabulary_counter = vocabulary_counter.most_common(vocabulary_limit - len(special_tokens))
        else:
            vocabulary_counter = vocabulary_counter.most_common()

        self.vocabulary = special_tokens + [token for token, _ in vocabulary_counter]
        self.token_ids = {token: index for index, token in enumerate(self.vocabulary)}

    def flatten_dialogs(self, dialogs):
        all_words = []
        for dialog in dialogs:
            for utterance in dialog:
                all_words.extend(utterance)
        return all_words

    def limit_dialogs_to_vocabulary(self, dialogs):
        for d_i, dialog in enumerate(dialogs):
            for u_i, utterance in enumerate(dialog):
                for t_i, token in enumerate(utterance):
                    if token not in self.vocabulary:
                        dialogs[d_i][u_i][t_i] = self.UNK

    def utterance_to_ids(self, utterance):
        utterance_ids = []

        for token in utterance:
            utterance_ids.append(self.token_ids.get(token, self.token_ids[self.UNK]))

        return utterance_ids

    def dialogs_to_ids(self, data):
        data_ids = []

        for dialog in data:
            dialog_ids = []

            for utterance in dialog:
                dialog_ids.append(self.utterance_to_ids(utterance))
            data_ids.append(dialog_ids)

        return data_ids

    def ids_to_tokens(self, ids):
        padding_id = self.token_ids[self.PAD]
        return [self.vocabulary[id] for id in ids if id != padding_id]

    def token_to_id(self, token):
        return self.token_ids[token]

    def get_train_dataset(self, context_size=2, min_reply_length=None, max_reply_length=None):
        return self.get_dataset(self.train_dialogs, context_size, min_reply_length, max_reply_length)

    def get_validation_dataset(self, context_size=2, min_reply_length=None, max_reply_length=None):
        return self.get_dataset(self.validation_dialogs, context_size, min_reply_length, max_reply_length)

    def get_test_dataset(self, context_size=2, min_reply_length=None, max_reply_length=None):
        return self.get_dataset(self.test_dialogs, context_size, min_reply_length, max_reply_length)

    def get_dataset(self, dialogs, context_size, min_reply_length, max_reply_length):
        dialogs_ids = self.dialogs_to_ids(dialogs)
        return DPDataset(self, dialogs_ids, context_size, min_reply_length, max_reply_length)

    def get_collator(self, reply_length=None):
        return DPCollator(self.token_ids[self.PAD], reply_length=reply_length)
