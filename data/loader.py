import os
import sys
import random
import json
import pandas
import pickle
import numpy as np
from tqdm import tqdm

from data.processor import PodcastEpisode

import torch
from transformers import BartTokenizer

CONFIDENCE_PATH = "/home/alta/summary/pm574/podcast_sum0/lib/processed_data/confidence.tsv"
def load_podcast_data(sets):
    podcasts = []
    if sets == -1:
        sets = [x for x in range(10)]
    for i in sets:
        path  = "/home/alta/summary/pm574/podcast_sum0/lib/data/podcast_set{}.bin".format(i)
        with open(path, 'rb') as f:
            set_of_podcasts = pickle.load(f, encoding="bytes")
        podcasts.extend(set_of_podcasts)
        print('loaded:', path)
    return podcasts

def load_podcast_random_filtered_data(sets):
    podcasts = []
    if sets == -1:
        sets = [x for x in range(10)]
    for i in sets:
        path  = "/home/alta/summary/pm574/podcast_sum0/lib/data/filtered_random_train/podcast_set{}.bin".format(i)
        with open(path, 'rb') as f:
            set_of_podcasts = pickle.load(f, encoding="bytes")
        podcasts.extend(set_of_podcasts)
        print('loaded:', path)
    return podcasts

def load_podcast_hier30k_filtered_data(sets):
    podcasts = []
    if sets == -1:
        sets = [x for x in range(10)]
    for i in sets:
        path  = "/home/alta/summary/pm574/podcast_sum0/lib/data/filtered_hier30k_train/podcast_set{}.bin".format(i)
        with open(path, 'rb') as f:
            set_of_podcasts = pickle.load(f, encoding="bytes")
        podcasts.extend(set_of_podcasts)
        print('loaded:', path)
    return podcasts

def load_podcast_hier30k_1040_filtered_data(sets):
    podcasts = []
    if sets == -1:
        sets = [x for x in range(10)]
    for i in sets:
        path  = "/home/alta/summary/pm574/podcast_sum0/lib/data/filtered_hier30k_train1040/podcast_set{}.bin".format(i)
        with open(path, 'rb') as f:
            set_of_podcasts = pickle.load(f, encoding="bytes")
        podcasts.extend(set_of_podcasts)
        print('loaded:', path)
    return podcasts

# using ROUGE-2 Recall to extract salient sentences (using original order for ties)
def load_podcast_oracle_1000_filtered_data(sets):
    podcasts = []
    if sets == -1:
        sets = [x for x in range(10)]
    for i in sets:
        path  = "/home/alta/summary/pm574/podcast_sum0/lib/data/filtered_oracle_train1000/podcast_set{}.bin".format(i)
        with open(path, 'rb') as f:
            set_of_podcasts = pickle.load(f, encoding="bytes")
        podcasts.extend(set_of_podcasts)
        print('loaded:', path)
    return podcasts

def load_podcast_oraclev3_1000_filtered_data(sets):
    podcasts = []
    if sets == -1:
        sets = [x for x in range(10)]
    for i in sets:
        path  = "/home/alta/summary/pm574/podcast_sum0/lib/data/filtered_oraclev3_train1000/podcast_set{}.bin".format(i)
        with open(path, 'rb') as f:
            set_of_podcasts = pickle.load(f, encoding="bytes")
        podcasts.extend(set_of_podcasts)
        print('loaded:', path)
    return podcasts

# using ROUGE-2 Recall (but no padding, e.g. only sentences with positive recall)
def load_podcast_oracler2_filtered_data(sets):
    podcasts = []
    if sets == -1:
        sets = [x for x in range(10)]
    for i in sets:
        path  = "/home/alta/summary/pm574/podcast_sum0/lib/data/filtered_oracler2_train/podcast_set{}.bin".format(i)
        with open(path, 'rb') as f:
            set_of_podcasts = pickle.load(f, encoding="bytes")
        podcasts.extend(set_of_podcasts)
        print('loaded:', path)
    return podcasts

# using ROUGE-2 F1 (Greedily searched for ORACLE) to extract salient sentences - nothing found => choose top3 longest.
def load_podcast_oraclef1_filtered_data(sets):
    podcasts = []
    if sets == -1:
        sets = [x for x in range(10)]
    for i in sets:
        path  = "/home/alta/summary/pm574/podcast_sum0/lib/data/filtered_oraclef1_train_v1/podcast_set{}.bin".format(i)
        with open(path, 'rb') as f:
            set_of_podcasts = pickle.load(f, encoding="bytes")
        podcasts.extend(set_of_podcasts)
        print('loaded:', path)
    return podcasts

def load_brass_set_ids():
    with open("/home/alta/summary/pm574/data/spotify-podcasts/summarisation-task-brass-set/filtered-episode-ids.txt", 'r') as f:
        lines = f.readlines()
    ids = [None for _ in range(len(lines))]
    for i, line in enumerate(lines):
        id = int(line.strip())
        ids[i] = id
    return ids

def load_dev150_set_ids():
    with open("/home/alta/summary/pm574/data/spotify-podcasts/no-audio/spotify-podcasts-2020/dev-set/150-episode-ids.txt", 'r') as f:
        lines = f.readlines()
    ids = [None for _ in range(len(lines))]
    for i, line in enumerate(lines):
        id = int(line.strip())
        ids[i] = id
    return ids

class BartBatcher(object):
    def __init__(self, tokenizer, config, podcasts, torch_device):
        self.cur_id    = 0
        self.epoch_counter = 0
        self.max_count = len(podcasts)
        self.device    = torch_device

        self.tokenizer = tokenizer
        self.podcasts  = podcasts
        self.config    = config
        # Google ASR confidence at episode level
        self.confidence = pandas.read_csv(CONFIDENCE_PATH, sep='\t', names=['id','score'], header=None)

        self.brass_set_ids = load_brass_set_ids()
        self.dev150_set_ids = load_dev150_set_ids()


    def shuffle_podcasts(self):
        random.shuffle(self.podcasts)
        return

    def is_podcast_good(self, id):
        podcast_id = self.podcasts[id].podcast_id
        if podcast_id in self.brass_set_ids and podcast_id not in self.dev150_set_ids and len(self.podcasts[id].description.split()) >= 5:
            return True
        else:
            return False



    def increment_pod_id(self):
        self.cur_id += 1
        if self.cur_id == self.max_count:
            self.cur_id = 0
            self.shuffle_podcasts()
            self.epoch_counter += 1
        return

    def store_predicted_grades(self, path):
        with open(path, 'rb') as f:
            grades = pickle.load(f, encoding="bytes")
        self.predicted_grades = grades
        print("loaded predicted grades:", path)
        return

    def get_a_batch(self, batch_size, pad_to_max_length=True):
        batch_count = 0
        inputs  = [None for _ in range(batch_size)]
        targets = [None for _ in range(batch_size)]

        while batch_count < batch_size:
            if not self.is_podcast_good(self.cur_id):
                self.increment_pod_id()
                continue

            inputs[batch_count]  = self.podcasts[self.cur_id].transcription
            targets[batch_count] = self.podcasts[self.cur_id].description

            self.increment_pod_id()
            batch_count += 1

        batch_encoded_inputs = self.tokenizer.batch_encode_plus(inputs,
            add_special_tokens=True, pad_to_max_length=True,
            max_length=self.config.max_position_embeddings, return_tensors='pt')

        input_ids      = batch_encoded_inputs['input_ids']
        attention_mask = batch_encoded_inputs['attention_mask']

        batch_encoded_targets = self.tokenizer.batch_encode_plus(targets,
            add_special_tokens=True, pad_to_max_length=pad_to_max_length,
            max_length=144, return_tensors='pt')
        # bos_token_id = 0
        target_ids = batch_encoded_targets['input_ids']
        target_attention_mask = batch_encoded_targets['attention_mask']


        if self.device == 'cuda':
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            target_ids = target_ids.to(self.device)
            target_attention_mask = target_attention_mask.to(self.device)

        return input_ids, attention_mask, target_ids, target_attention_mask

    def shifted_target_left(self, target_ids, target_attention_mask):
        # shifted LEFT
        shifted_target_ids = torch.zeros(target_ids.shape, dtype=target_ids.dtype)
        shifted_target_attention_mask = torch.zeros(target_attention_mask.shape, dtype=torch.float)
        shifted_target_ids[:,:-1] = target_ids.clone().detach()[:,1:]
        shifted_target_attention_mask[:,:-1] = target_attention_mask.clone().detach()[:,1:]

        if self.device == 'cuda':
            shifted_target_ids = shifted_target_ids.to(self.device)
            shifted_target_attention_mask = shifted_target_attention_mask.to(self.device)

        return shifted_target_ids, shifted_target_attention_mask

if __name__ == "__main__":
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BartTokenizer.from_pretrained('bart-large')
    batcher = BartBatcher(tokenizer, load_podcast_data(2), torch_device)
