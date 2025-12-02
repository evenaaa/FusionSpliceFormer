#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/28 17:23
# @Author  : even
# @File    : seq_tokenize.py
from collections import Counter
import json
import pickle
import os

class Tokenizer:
    def __init__(self, max_len=None, max_k=3):
        self.max_len = max_len
        self.max_k = max_k
        self.k_word2id = {k: {'<pad>': 0, '<unk>': 1} for k in range(1, max_k + 1)}
        self.k_id2word = {k: {0: '<pad>', 1: '<unk>'} for k in range(1, max_k + 1)}

    def token(self, texts):
        tokenized_texts = []
        for text in texts:
            tokens_for_k = {}
            for k in range(1, self.max_k + 1):
                # Overlapping k-mer segmentation
                tokens = [text[i:i + k] for i in range(len(text) - k + 1)]
                tokens_for_k[k] = tokens
            tokenized_texts.append(tokens_for_k)

        return tokenized_texts

    def fit_on_texts(self, texts):
        tokenized_texts = self.token(texts)
        for k in range(1, self.max_k + 1):
            token_list = []
            for tokens_dict in tokenized_texts:
                token_list += tokens_dict[k]
            word_counts = Counter(token_list)
            # Sort by frequency in descending order
            sorted_tokens = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            for token, count in sorted_tokens:
                if token not in self.k_word2id[k]:
                    self.k_word2id[k][token] = len(self.k_word2id[k])
                    self.k_id2word[k][self.k_word2id[k][token]] = token

    def texts_to_sequences(self, texts):
        tokenized_texts = self.token(texts)
        sequences_per_k = {k: [] for k in range(1, self.max_k + 1)}
        for tokens_dict in tokenized_texts:
            for k in range(1, self.max_k + 1):
                seq = [self.k_word2id[k].get(token, self.k_word2id[k]['<unk>']) for token in tokens_dict[k]]
                # If max_len is set, truncation or padding will be performed.
                if self.max_len is not None:
                    if len(seq) > self.max_len:
                        seq = seq[:self.max_len]
                    else:
                        pad_num = self.max_len - len(seq)
                        seq = seq + [self.k_word2id[k]['<pad>']] * pad_num
                sequences_per_k[k].append(seq)
        return sequences_per_k

    def save_vocab(self, word2id_file_prefix, id2word_file_prefix):
        for k in range(1, self.max_k + 1):
            with open(f"{word2id_file_prefix}_k{k}_word2id.json", "w", encoding="utf-8") as f:
                json.dump(self.k_word2id[k], f, ensure_ascii=False, indent=1)
            with open(f"{id2word_file_prefix}_k{k}_id2word.json", "w", encoding="utf-8") as f:
                json.dump(self.k_id2word[k], f, ensure_ascii=False, indent=1)
        print('word2id and id2word saved successfully.')

    def load_vocab(self, word2id_file_prefix, id2word_file_prefix):
        for k in range(1, self.max_k + 1):
            with open(f"{word2id_file_prefix}_k{k}_word2id.json", "r", encoding="utf-8") as f:
                self.k_word2id[k] = json.load(f)
            with open(f"{id2word_file_prefix}_k{k}_id2word.json", "r", encoding="utf-8") as f:
                self.k_id2word[k] = json.load(f)
        print('word2id and id2word loaded successfully.')

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'create dir\t{dir}')

def load_tokenizers(file_path):
    # Load the saved tokenizers using pickle
    with open(file_path, 'rb') as f:
        tokenizers = pickle.load(f)

    # Restore the tokenizer object
    tokenizer_ref = tokenizers['tokenizer_ref']
    tokenizer_alt = tokenizers['tokenizer_alt']

    return tokenizer_ref, tokenizer_alt

def make_dataloader(tokenizers_path,df):
    refseq = df['base400_refseq'].tolist()
    altseq = df['base400_altseq'].tolist()
    tokenizer_ref, tokenizer_alt=load_tokenizers(tokenizers_path)

    ref_ids = tokenizer_ref.texts_to_sequences(refseq)
    alt_ids = tokenizer_alt.texts_to_sequences(altseq)

    return ref_ids, alt_ids

def seq_tokenize(
        feature_df,
        seq_path,
        tokenizers_path='dataset/modeldata',
):

    acceptor_ref_ids, acceptor_alt_ids=make_dataloader(
        tokenizers_path=f'{tokenizers_path}/acceptor/tokenizers.pkl',
        df=feature_df[feature_df['region3']=='acceptor']
    )
    donor_ref_ids, donor_alt_ids=make_dataloader(
        tokenizers_path=f'{tokenizers_path}/donor/tokenizers.pkl',
        df=feature_df[feature_df['region3']=='donor']
    )
    exonic_ref_ids, exonic_alt_ids=make_dataloader(
        tokenizers_path=f'{tokenizers_path}/exonic/tokenizers.pkl',
        df=feature_df[feature_df['region3']=='exonic']
    )

    seq_ids_dict={
        'acceptor_ref_ids':acceptor_ref_ids,
        'acceptor_alt_ids':acceptor_alt_ids,
        'donor_ref_ids':donor_ref_ids,
        'donor_alt_ids':donor_alt_ids,
        'exonic_ref_ids':exonic_ref_ids,
        'exonic_alt_ids':exonic_alt_ids
    }
    json.dump(seq_ids_dict, open(seq_path, 'w', encoding='utf-8'), ensure_ascii=False)
