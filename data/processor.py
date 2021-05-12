import os
import sys
import re
import random
import json
import pandas
import pickle
from nltk import tokenize
from tqdm import tqdm

# PATH TO DOWNLOADED DATA
METADATA_PATH = "/home/alta/summary/pm574/data/spotify-podcasts/no-audio/spotify-podcasts-2020/metadata.tsv"
TRANSCRIPT_MAIN_PATH = "/home/alta/summary/pm574/data/spotify-podcasts/no-audio/spotify-podcasts-2020/podcasts-transcripts"

"""
The data will be put in directories:
    1) lib/data/episode_transcript
        - set0: id0 - id9999
        - set1: id10000 - id19999
        ...
        - set10: id100000 - id105359
    2) lib/data/episode_description
        - set0: id0 - id9999
        - set1: id10000 - id19999
        ...
        - set10: id100000 - id105359
    one line per file
"""

class PodcastEpisode(object):
    def __init__(self, podcast_id, transcription, description):
        self.podcast_id = podcast_id
        self.transcription = transcription
        self.description = description

def parse_json(json_path):
    with open(json_path, 'r') as file: x = file.read()
    y = json.loads(x)
    num_results = len(y['results'])
    transcripts = []
    for i in range(num_results - 1):
        if len(y['results'][i]['alternatives']) != 1: raise Exception("num alternative error")
        if y['results'][i]['alternatives'][0] != {}:
            transcript = y['results'][i]['alternatives'][0]['transcript']
            transcripts.append(transcript)
    episode_transcript = "".join(transcripts)
    return episode_transcript

# ----------------------------------------------------------------------------------- #
def containURL(sentence):
    x = re.search("((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*", sentence)
    if x == None: return False
    else: return True
def containAt(sentence):
    if '@' in sentence: return True
    else: return False
def containAnchor(sentence):
    if "Anchor: The easiest way to make a podcast." in sentence:
        return True
    else:
        return False

def clean_episode_description(txt):
    filtered = []
    sentences = tokenize.sent_tokenize(txt)
    for i, sent in enumerate(sentences):
        # if not containURL(sent) and not containAt(sent) and not containAnchor(sent):
        if not containURL(sent) and not containAnchor(sent):
            filtered.append(sent)
    filtered_description = " ".join(filtered)
    return filtered_description
# ----------------------------------------------------------------------------------- #

def start(start_id, end_id, podcast_out_path):
    """
        Example:
        start_id = 0
        end_id   = 10000
        podcast_out_path  = /home/alta/summary/pm574/podcast_sum0/lib/data/podcast_set0.bin
    """

    metadata = pandas.read_csv(METADATA_PATH, sep='\t')
    podcasts  = [None for _ in range(end_id-start_id)]
    print(podcast_out_path)
    print("id = [{},{})".format(start_id, end_id))

    for jj, id in tqdm(enumerate(range(start_id, end_id))):
        show_id          = metadata['show_uri'][id].split(':')[-1]
        show_pf          = metadata['show_filename_prefix'][id].split(':')[-1]
        episode_pf       = metadata['episode_filename_prefix'][id]

        episode_description = metadata['episode_description'][id]
        if isinstance(episode_description, str):
            episode_description = clean_episode_description(episode_description)
        else:
            episode_description = '.'

        trascript_path      = "{}/{}/{}/{}/{}.json".format(TRANSCRIPT_MAIN_PATH, show_id[0].upper(), show_id[1].upper(), show_pf, episode_pf)
        episode_transcript  = parse_json(trascript_path)

        podcasts[jj] = PodcastEpisode(podcast_id=id, transcription=episode_transcript, description=episode_description)

    with open(podcast_out_path, "wb") as f:
        pickle.dump(podcasts, f)

if __name__ == "__main__":
    if(len(sys.argv) == 4):
        # start,end = (0,10000), (10000,20000), ..., (100000, 105360) => set0 to set10 where set0-set9=train | set10=valid
        start_id = int(sys.argv[1])
        end_id   = int(sys.argv[2])
        save_path = sys.argv[3]
        start(start_id, end_id, save_path)
    else:
        print("Usage: python processor.py start_id end_id save_path")
        raise Exception("argv error")
