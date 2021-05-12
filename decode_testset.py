import os
import sys
import pickle
import random
from datetime import datetime
from collections import OrderedDict

import torch
from transformers import BartTokenizer, BartForConditionalGeneration

from data.processor import PodcastEpisode
from nltk import tokenize

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("torch_device:", torch_device)

MODEL_NAME  = "bartbaseline-1024-NOV23-v1-step160000"             # Truncate
DATA_PATH   = "/home/alta/summary/pm574/podcast_sum0/lib/test_data/podcast_testset.bin"
# -------------------------------------------------- #

TRAINED_MODEL_PATH = "/home/alta/summary/pm574/podcast_sum0/lib/trained_models2/{}.pt".format(MODEL_NAME)
DECODE_DIR         = "/home/alta/summary/pm574/podcast_sum0/system_output/{}/{}".format(MODEL_NAME, "testset")

print("MODEL_NAME:", MODEL_NAME)
print("DATA_PATH:", DATA_PATH)
print("DECODE_DIR:", DECODE_DIR)

def decode(start_id, end_id):
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    # ---------------------------------- Model ---------------------------------- #
    # Bart Vanilla
    bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    # --------------------------------------------------------------------------- #

    if torch_device == 'cuda':
        bart.cuda()
        state = torch.load(TRAINED_MODEL_PATH)
    else:
        state = torch.load(TRAINED_MODEL_PATH, map_location=torch.device('cpu'))

    model_state_dict = state['model']
    bart.load_state_dict(model_state_dict)

    # data
    print("DATA_PATH =", DATA_PATH)
    with open(DATA_PATH, 'rb') as f:
        podcasts = pickle.load(f, encoding="bytes")
    print("len(podcasts) = {}".format(len(podcasts)))

    ids = [x for x in range(start_id, end_id)]
    random.shuffle(ids)

    bart.eval() # not needed but for sure!!

    for id in ids:
        # check if the file exist or not
        out_path = "{}/{}_decoded.txt".format(DECODE_DIR, id)
        exist = os.path.isfile(out_path)
        if exist:
            print("id {}: already exists".format(id))
            continue

        article_input_ids = bart_tokenizer.batch_encode_plus([podcasts[id].transcription],
            return_tensors='pt', max_length=bart.config.max_position_embeddings)['input_ids'].to(torch_device)
        summary_ids = bart.generate(article_input_ids,
                        num_beams=4, length_penalty=2.0,
                        max_length=144, # set this equal to the max length in training
                        min_length=56,  # one sentence
                        no_repeat_ngram_size=3)

        summary_txt = bart_tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True).strip()
        with open(out_path, 'w') as file:
            file.write(summary_txt)
        print("write:", out_path)

def write_reference():
    # data
    data_path = "/home/alta/summary/pm574/podcast_sum0/lib/test_data/podcast_testset.bin"
    with open(data_path, 'rb') as f:
        podcasts = pickle.load(f, encoding="bytes")
    print("len(podcasts) = {}".format(len(podcasts)))

    for i in range(len(podcasts)):
        summary = podcasts[i].description
        sentences = tokenize.sent_tokenize(summary.strip())
        out_path = "/home/alta/summary/pm574/podcast_sum0/reference/testset/{}_reference.txt".format(i)
        with open(out_path, 'w') as file:
            file.write(" ".join(sentences))
            # file.write("\n".join(sentences))
        print("write:", out_path)

if __name__ == "__main__":
    # To get the reference (i.e. creator-provided) summaries, uncomment this line
    # It's for computing, ROUGE scores, but just run it once!
    # write_reference()

    if(len(sys.argv) == 2):
        start_id = int(sys.argv[1])
        end_id   = start_id + 10
        if end_id > 1027: end_idx = 1027
        decode(start_id, end_id)
    elif(len(sys.argv) == 3):
        start_id = int(sys.argv[1])
        end_id   = int(sys.argv[2])
        decode(start_id, end_id)
    else:
        print("Usage: python decode_testset.py start_id end_id")
        raise Exception("argv error")
