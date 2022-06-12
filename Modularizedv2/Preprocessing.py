# Base Packages
import os
import spacy
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from keras.preprocessing.sequence import pad_sequences


class AmazonReviewsDataset ():
    def __init__(self, df, max_len=80, frequency_threshold=2,
                 spacy_tokenizer=spacy.load('en_core_web_sm'),
                 vocab=None) -> None:

        self.spacy_tokenizer = spacy_tokenizer
        self.frequency_threshold = frequency_threshold

        sentences = df['text'].map(self.tokenize_sentence, na_action='ignore')
        if vocab is None:
            self.stoi, self.itos = self.build_vocab(sentences)
        else:
            self.stoi, self.itos = vocab

        encoded_sentences = sentences.map(self.int_encode)

        self.src = pad_sequences(encoded_sentences,
                                 padding='post',
                                 truncating="post",
                                 maxlen=max_len,
                                 value=0)

        self.trg = self.build_int_targets(df['label'])

    def int_encode(self, sentence):
        return [self.stoi[word] if word in self.stoi.keys() else 1 for word in sentence]

    def tokenize_sentence(self, sentence):
        return [token.text.lower() for token in self.spacy_tokenizer.tokenizer(sentence)]

    def build_vocab(self, sentences):
        frequencies = {}
        stoi = {'<pad>': 0, '<unk>': 1}
        itos = {v: k for k, v in stoi.items()}

        for sentence in sentences:
            for word in sentence:
                if word in frequencies:
                    frequencies[word] += 1
                else:
                    frequencies[word] = 1

        sorted_frequencies = dict(
            sorted(frequencies.items(), key=lambda item: item[1], reverse=True))

        idx = len(stoi)
        for word, freq in sorted_frequencies.items():
            if word not in stoi and freq >= self.frequency_threshold:

                stoi[word] = idx
                itos[idx] = word

                idx += 1

        return (stoi, itos)

    def build_int_targets(self, labels):
        targets = labels.map(lambda x: np.array(1.0) if x == '__label__1' else np.array(
            0.0) if x == '__label__2' else np.array(np.nan))
        assert not np.nan in targets

        return targets.tolist()

    @staticmethod
    def get_output_dim():
        return 1

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, idx):
        return {
            'src': self.src[idx],
            'trg': self.trg[idx]
        }

    @staticmethod
    def initialize_data(csv_files, dataset_percent):
        df = pd.concat([
            pd.read_csv(csv_file, delimiter='\n', names=['all'])
            for csv_file in csv_files
        ], ignore_index=True)

        df[['label', 'text']] = df['all'].str.split(' ', expand=True, n=1)
        df.drop(columns=['all'], inplace=True)

        df = df.iloc[:int(len(df)*dataset_percent)].copy()  # Normally 0.1

        return df


class IMBD_Dataset ():
    def __init__(self, df, max_len=80, frequency_threshold=2,
                 spacy_tokenizer=spacy.load('en_core_web_sm'),
                 vocab=None) -> None:

        self.spacy_tokenizer = spacy_tokenizer
        self.frequency_threshold = frequency_threshold

        sentences = df['text'].map(self.tokenize_sentence, na_action='ignore')
        if vocab is None:
            self.stoi, self.itos = self.build_vocab(sentences)
        else:
            self.stoi, self.itos = vocab

        encoded_sentences = sentences.map(self.int_encode)

        self.src = pad_sequences(encoded_sentences,
                                 padding='post',
                                 truncating="post",
                                 maxlen=max_len,
                                 value=0)

        self.trg = self.build_int_targets(df['label'])

    def int_encode(self, sentence):
        return [self.stoi[word] if word in self.stoi.keys() else 1 for word in sentence]

    def tokenize_sentence(self, sentence):
        return [token.text.lower() for token in self.spacy_tokenizer.tokenizer(sentence)]

    def build_vocab(self, sentences):
        frequencies = {}
        stoi = {'<pad>': 0, '<unk>': 1}
        itos = {v: k for k, v in stoi.items()}

        for sentence in sentences:
            for word in sentence:
                if word in frequencies:
                    frequencies[word] += 1
                else:
                    frequencies[word] = 1

        sorted_frequencies = dict(
            sorted(frequencies.items(), key=lambda item: item[1], reverse=True))

        idx = len(stoi)
        for word, freq in sorted_frequencies.items():
            if word not in stoi and freq >= self.frequency_threshold:

                stoi[word] = idx
                itos[idx] = word

                idx += 1

        return (stoi, itos)

    def build_int_targets(self, labels):
        targets = labels.map(lambda x: np.array(1.0) if x == 'positive' else np.array(
            0.0) if x == 'negative' else np.array(np.nan))
        assert not np.nan in targets

        return targets.tolist()

    @staticmethod
    def get_output_dim():
        return 1

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, idx):
        return {
            'src': self.src[idx],
            'trg': self.trg[idx]
        }

    @staticmethod
    def initialize_data(csv_files):
        df = pd.concat([
            pd.read_csv(csv_file, names=['text', 'label'], header=0)
            for csv_file in csv_files
        ], ignore_index=True)

        ####### REMOVE SAMPLING######
        # df = df.copy().sample(5000, random_state=1).reset_index(drop=True)
        df = df.iloc[:int(len(df)*0.1)].copy().reset_index(drop=True)

        return df
