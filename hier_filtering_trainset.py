"""
1. This script is used for both train and dev sets (e.g. SPFT_train (set0,...set9), SPFT_dev (set10, set_dev1)
2. Decode first then Combine (see commented section below)
3. For the dev set, to go from set_dev1 to set_dev1_brass please use data/set_dev1_to_brass.py

@potsawee 25 Nov 2020
"""

import os
import sys
import pickle
import random
import torch
import numpy as np
from nltk import tokenize
from tqdm import tqdm

from data.loader import BartBatcher, load_podcast_data
from data.processor import PodcastEpisode
from hier_model import Batch, HierTokenizer, HierarchicalModel

from transformers import BartTokenizer

if torch.cuda.is_available():
    torch_device = 'cuda'
    use_gpu = True
else:
    torch_device = 'cpu'
    use_gpu = False

PODCAST_SET  = 10 # train=0,1,2,3,...,9, valid=10
DATA_PATH    = "/home/alta/summary/pm574/podcast_sum0/lib/data/podcast_set{}.bin".format(PODCAST_SET)

MAX_BART_LEN   = 1024
MAX_INPUT_SENT = 1000
MAX_SENT_WORD  = 50
HIER_MODEL   = "SPOTIFY_long"
MODEL_STEP   = 30000

print("PODCAST_SET:", PODCAST_SET)
print("HIER_MODEL:", HIER_MODEL)
print("MODEL_STEP:", MODEL_STEP)

def filtering_data(start_id, end_id):
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    with open(DATA_PATH, 'rb') as f:
        podcasts = pickle.load(f, encoding="bytes")
    print("len(podcasts) = {}".format(len(podcasts)))

    hier_tokenizer = HierTokenizer()
    hier_tokenizer.set_len(num_utterances=MAX_INPUT_SENT, num_words=MAX_SENT_WORD)
    hier_model = HierarchicalModel(HIER_MODEL, use_gpu=use_gpu)

    ids = [x for x in range(start_id, end_id)]
    random.shuffle(ids)

    for i in ids:
        # check if the file exist or not
        # DECODER_DIR = temp folder
        out_path = "/home/alta/summary/pm574/podcast_sum0/lib/data/filtered_hier30k_train{}/decode{}/{}_filtered_transcription.txt".format(MAX_BART_LEN, PODCAST_SET, i)
        exist = os.path.isfile(out_path)
        if exist:
            print("id {}: already exists".format(i))
            continue

        l1 = len(bart_tokenizer.encode(podcasts[i].transcription, max_length=50000))

        if l1 < MAX_BART_LEN:
            filtered_transcription = podcasts[i].transcription

        else:
            sentences = tokenize.sent_tokenize(podcasts[i].transcription)
            keep_idx = []
            batch = hier_tokenizer.get_enc_input([podcasts[i].transcription], use_gpu=use_gpu)[0]

            # This must only be done at training time - as it replies on the target to get attention
            # target, tgt_len = hier_tokenizer.get_dec_target([podcasts[i].description], max_len=300, use_gpu=use_gpu)
            # attention  = hier_model.get_utt_attn_with_ref(batch, target, tgt_len)

            attention = hier_model.get_utt_attn_without_ref(batch, beam_width=4, time_step=144, penalty_ug=0.0, alpha=1.25, length_offset=5)


            if len(sentences) != attention.shape[0]:
                if len(sentences) > MAX_INPUT_SENT:
                    sentences = sentences[:MAX_INPUT_SENT]
                else:
                    raise ValueError("shape error #1")

            selection_score = attention

            rank = np.argsort(selection_score)[::-1]
            keep_idx = []
            total_length = 0
            for sent_i in rank:
                if total_length < MAX_BART_LEN:
                    sent = sentences[sent_i]
                    length = len(bart_tokenizer.encode(sent)[1:-1]) # ignore <s> and </s>
                    total_length += length
                    keep_idx.append(sent_i)
                else:
                    break

            keep_idx = sorted(keep_idx)
            # for sent_i in keep_idx:
            #     print("sent{} [{:.3f}]: {}".format(sent_i, 100*attention[sent_i], sentences[sent_i]))
            filtered_sentences = [sentences[j] for j in keep_idx]
            filtered_transcription = " ".join(filtered_sentences)

        with open(out_path, "w") as f:
            f.write(filtered_transcription)

        print("write:", out_path)


def combine():
    with open(DATA_PATH, 'rb') as f:
        podcasts = pickle.load(f, encoding="bytes")
    print("len(podcasts) = {}".format(len(podcasts)))

    for i in tqdm(range(len(podcasts))):
        out_path = "/home/alta/summary/pm574/podcast_sum0/lib/data/filtered_hier30k_train{}/decode{}/{}_filtered_transcription.txt".format(MAX_BART_LEN, PODCAST_SET, i)

        with open(out_path, 'r') as f:
            x = f.read()
        podcasts[i].transcription = x

    save_filtered_data_path = "/home/alta/summary/pm574/podcast_sum0/lib/data/filtered_hier30k_train{}/podcast_set{}.bin".format(MAX_BART_LEN, PODCAST_SET)
    with open(save_filtered_data_path, "wb") as f:
        pickle.dump(podcasts, f)

if __name__ == "__main__":
    # once decoding (i.e. filtering_data) is done, combine them using combine()
    # combine()

    if(len(sys.argv) == 2):
        start_id = int(sys.argv[1])
        end_id   = start_id + 50 # 5360 / 10 = 536
        # end_id   = start_id + 100 # 10000 / 100 = 100
        # if end_id > 10000: end_idx = 10000
        if end_id > 5360: end_idx = 5360
        filtering_data(start_id, end_id)
    elif(len(sys.argv) == 3):
        start_id = int(sys.argv[1])
        end_id   = int(sys.argv[2])
        filtering_data(start_id, end_id)
    else:
        print("Usage: python filtering_data.py start_id end_id")
        raise Exception("argv error")
