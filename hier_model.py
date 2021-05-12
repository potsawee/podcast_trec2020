import os
from collections import OrderedDict
from nltk import tokenize
import torch
import numpy as np

from hierarchical_rnn_v5 import EncoderDecoder
from transformers import BertTokenizer

if torch.__version__ == '1.2.0': KEYPADMASK_DTYPE = torch.bool
else:
    print("source ~/anaconda3/bin/activate torch12-cuda10")
    raise Exception("Torch Version not supported")

LONG_BORING_TENNIS_ARTICLE = """
 Andy Murray  came close to giving himself some extra preparation time for his w
edding next week before ensuring that he still has unfinished tennis business to
 attend to. The world No 4 is into the semi-finals of the Miami Open, but not be
fore getting a scare from 21 year-old Austrian Dominic Thiem, who pushed him to
4-4 in the second set before going down 3-6 6-4, 6-1 in an hour and three quarte
rs. Murray was awaiting the winner from the last eight match between Tomas Berdy
ch and Argentina's Juan Monaco. Prior to this tournament Thiem lost in the secon
d round of a Challenger event to soon-to-be new Brit Aljaz Bedene. Andy Murray p
umps his first after defeating Dominic Thiem to reach the Miami Open semi finals
 . Muray throws his sweatband into the crowd after completing a 3-6, 6-4, 6-1 vi
ctory in Florida . Murray shakes hands with Thiem who he described as a 'strong
guy' after the game . And Murray has a fairly simple message for any of his fell
ow British tennis players who might be agitated about his imminent arrival into
the home ranks: don't complain. Instead the British No 1 believes his colleagues
 should use the assimilation of the world number 83, originally from Slovenia, a
s motivation to better themselves. At present any grumbles are happening in priv
ate, and Bedene's present ineligibility for the Davis Cup team has made it less
of an issue, although that could change if his appeal to play is allowed by the
International Tennis Federation. Murray thinks anyone questioning the move, now
it has become official, would be better working on getting their ranking closer
to his. 'If he was 500 in the world they wouldn't be that fussed about it but ob
viously he threatens their position a bit,' said the 27 year-old Scot. ' and he'
s obviously the British number two, comfortably. 'So they can complain but the b
est thing to do is use it in the right way and accept it for what it is, and try
 to use it as motivation whether they agree with it or not. He's British now so
they've just got to deal with it. Murray stretches for a return after starting h
is quarter final match slowly on the show court . Thiem held nothing back as he
raced through the opening set, winning it 6-3 with a single break . The young Au
strian is considered to be one of the hottest prospects on the ATP Tour . 'I wou
ld hope that all the guys who are below him now like James (Ward) , Kyle (Edmund
) , Liam (Broady) they will use it as motivation. If he becomes eligible for Dav
is Cup then those guys are going to have to prove themselves. 'It can only be se
en as a positive for those guys using it to try to get better. He's a good playe
r but so are James and Kyle and Liam has improved. Aljaz is there, he's on the t
our every week, the other guys aren't quite there yet.' For the first time Murra
y, who has an encyclopaedic knowledge of the top 100, gave his opinion of Bedene
: 'He's a good player with a very good serve. He's a legitimate top 100 player,
when he plays Challengers he's there or thereabouts, when he plays on the main t
our he wins matches, it's not like he turns up and always loses in the first rou
nd. Murray's fiancee was once again watching from the stands shaded by a huge br
immed hat . Kim Sears flashes her enormous diamond engagement ring while watchin
g her beau on court . 'He had a bad injury last year (wrist) but has recovered w
ell. I would imagine he would keep moving up the rankings although I don't know
exactly how high he can go. I've practised with him a couple of times, I haven't
 seen him play loads, but when you serve as well as he does it helps. I would im
agine he' s going to be comfortably in the top 70 or 80 in the world for a while
.' It is understood the Lawn Tennis Association will give background support to
his case regarding the Davis Cup but have made it clear that the onus is on him
to lead the way. An official statement said: 'To have another player in the men'
s top 100 is clearly a positive thing for British tennis and so we very much wel
come Aljaz's change in citizenship.' The last comparable switch came twenty year
s ago when Greg Rusedski arrived from Canada. It was by no means universally pop
ular but, like Bedene, he pledged that he was in for the long haul and, in fairn
ess to him, he proved true to his word. Loising the first set shocked Murray int
o life as he raced to a commanding lead in the second . The No 3 seed sent over
a few glaring looks towards his team before winning the second set . Murray had
to put such matters aside as he tackled the unusually talented Thiem, a delight
to watch. Coached by Boris Becker's veteran mentor Gunter Bresnik, he slightly r
esembles Andy Roddick and hits with similar power but more elegance. His single
handed backhand is a thing of rare beauty. However, he has had a mediocre season
 coming into this event and there was little to forewarn of his glorious shotmak
ing that seemed to catch Murray unawares early on. The world No 4 looked to have
 worked him out in the second, but then suffered one of his periopdic mental lap
ses and let him back in from 4-1 before closing it out with a break. After break
ing him for 3-1 in the decider the Austrian whirlwind burnt itself out. 'He's a
strong guy who hits the ball hard and it became a very physical match,' said Mur
ray. Murray was presented with a celebratory cake after winning his 500th match
in the previous round .
""".replace('\n','')

class Batch(object):
    def __init__(self, input, u_len, w_len):
        self.input = input
        self.u_len = u_len
        self.w_len = w_len


class HierTokenizer(object):
    def __init__(self):
        self.num_utterances = 50    # max no. utterance in a meeting
        self.num_words      = 32    # max no. words in an utterance
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.SEP_TOKEN  = '[SEP]'
        self.STOP_TOKEN = '[MASK]'

    def set_len(self, num_utterances, num_words):
        self.num_utterances = num_utterances
        self.num_words      = num_words

    def get_enc_input(self, docs, use_gpu=False):
        batch_size = len(docs)
        input = np.zeros((batch_size, self.num_utterances, self.num_words), dtype=np.long)
        u_len = np.zeros((batch_size), dtype=np.long)
        w_len = np.zeros((batch_size, self.num_utterances), dtype=np.long)

        for i, doc in enumerate(docs):
            sentences = tokenize.sent_tokenize(doc)
            num_sentences = len(sentences)
            if num_sentences > self.num_utterances:
                num_sentences = self.num_utterances
                sentences = sentences[:self.num_utterances]
            u_len[i] = num_sentences

            for j, sent in enumerate(sentences):
                # BERT tokenizer (base-uncased) encode the same regardless of the case
                token_ids = self.bert_tokenizer.encode(sent)[1:-1] # remove [CLS], [SEP]
                utt_len = len(token_ids)
                if utt_len > self.num_words:
                    utt_len = self.num_words
                    token_ids = token_ids[:self.num_words]
                input[i,j,:utt_len] = token_ids
                w_len[i,j] = utt_len
        input = torch.from_numpy(input)
        u_len = torch.from_numpy(u_len)
        w_len = torch.from_numpy(w_len)

        if use_gpu:
            input = input.cuda()
            u_len = u_len.cuda()
            w_len = w_len.cuda()

        batches = [None for _ in range(batch_size)]
        for i in range(batch_size):
            batch = Batch(input[i:i+1,:,:], u_len[i:i+1], w_len[i:i+1,:])
            batches[i] = batch

        return batches

    def get_dec_target(self, summaries, max_len=300, use_gpu=False):
        batch_size = len(summaries)
        target  = np.zeros((batch_size, max_len), dtype=np.long)
        target.fill(103)
        tgt_len = np.zeros((batch_size), dtype=np.int)
        for i, summary in enumerate(summaries):
            concat_tokens = [101]
            sentences = tokenize.sent_tokenize(summary)
            for j, sent in enumerate(sentences):
                token_ids = self.bert_tokenizer.encode(sent)[1:-1] # remove [CLS], [SEP]
                concat_tokens.extend(token_ids)
                concat_tokens.extend([102]) # [SEP]
            tl = len(concat_tokens)
            if tl > max_len:
                concat_tokens = concat_tokens[:max_len]
                tl = max_len
            target[i, :tl] = concat_tokens
            tgt_len[i] = tl
        target = torch.from_numpy(target)
        if use_gpu:
            target = target.cuda()
        return target, tgt_len

    def tgtids2summary(self, tgt_ids):
        # tgt_ids = a row of numpy array containing token ids
        bert_decoded = self.bert_tokenizer.decode(tgt_ids)
        # truncate START_TOKEN & part after STOP_TOKEN
        stop_idx = bert_decoded.find(self.STOP_TOKEN)
        processed_bert_decoded = bert_decoded[5:stop_idx]
        summary = [s.strip() for s in processed_bert_decoded.split(self.SEP_TOKEN)]
        return summary

class HierarchicalModel(object):
    def __init__(self, model_name, model_step=None, use_gpu=False):
        if model_name == "SPOTIFY_long":
            self.model_path = "/home/alta/summary/pm574/podcast_sum0/lib/released_weights/HIERMODEL_640_50_step30000.pt"
            exist = os.path.isfile(self.model_path)
            if exist == False:
                raise Exception("Model Checkpoint ({}) does not exist".format(self.model_path))
            self.load_option = 2
            print("Hier Model checkpoint at:", self.model_path)

        else:
            self.model_path = model_name
            self.load_option = 2 # new version

        args = {}
        args['vocab_size']      = 30522 # BERT tokenizer
        args['embedding_dim']   = 256   # word embeeding dimension
        args['rnn_hidden_size'] = 512   # RNN hidden size
        args['dropout']        = 0.0
        args['num_layers_enc'] = 2
        args['num_layers_dec'] = 1
        args['memory_utt']     = False

        self.device = 'cuda' if use_gpu else 'cpu'
        self.model = EncoderDecoder(args, self.device)
        self.load_model()
        self.model.eval()

    def load_model(self):
        if self.device == 'cuda':
            try:
                state = torch.load(self.model_path)
                if self.load_option == 1:
                    self.model.load_state_dict(state)
                elif self.load_option == 2:
                    model_state_dict = state['model']
                    self.model.load_state_dict(model_state_dict)
                print("load succesful #1")
            except:
                if self.load_option == 1:
                    model_state_dict = torch.load(self.model_path)
                new_model_state_dict = OrderedDict()
                for key in model_state_dict.keys():
                    new_model_state_dict[key.replace("module.","")] = model_state_dict[key]
                self.model.load_state_dict(new_model_state_dict)
                print("load succesful #2")
        elif self.device == 'cpu':
            try:
                state = torch.load(self.model_path, map_location=torch.device('cpu'))
                if self.load_option == 1:
                    self.model.load_state_dict(state)
                elif self.load_option == 2:
                    model_state_dict = state['model']
                    self.model.load_state_dict(model_state_dict)
                print("load succesful #3")
            except:
                if self.load_option == 1:
                    model_state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))

                new_model_state_dict = OrderedDict()
                for key in model_state_dict.keys():
                    new_model_state_dict[key.replace("module.","")] = model_state_dict[key]
                self.model.load_state_dict(new_model_state_dict)
                print("load succesful #4")

    def decode(self, tokenizer, batches, beam_width=4, time_step=144, penalty_ug=0.0, alpha=1.25, length_offset=5):
        decode_dict = {
            'k': beam_width,
            'time_step': time_step,
            'vocab_size': 30522,
            'device': self.device,
            'start_token_id': 101, 'stop_token_id': 103,
            'alpha': alpha,
            'length_offset': length_offset,
            'penalty_ug': penalty_ug,
            'keypadmask_dtype': KEYPADMASK_DTYPE,
            'memory_utt': False,
            'batch_size': 1
        }
        summaries = [None for _ in range(len(batches))]
        for i, batch in enumerate(batches):
            summary_id = self.beam_search(batch, decode_dict)
            sentences = tokenizer.tgtids2summary(summary_id)
            summaries[i] = " ".join(sentences)
        return summaries

    def beam_search(self, batch, decode_dict):
        input = batch.input
        u_len = batch.u_len
        w_len = batch.w_len
        with torch.no_grad():
            summary_id, _, _ = self.model.decode_beamsearch(input, u_len, w_len, decode_dict)
        return summary_id

    def get_utt_attn_with_ref(self, enc_batch, target, tgt_len):
        # batch_size should be 1
        with torch.no_grad():
            # Teacher Forcing
            _, _, _, _, u_attn_scores = self.model(enc_batch.input, enc_batch.u_len, enc_batch.w_len, target)
        N = enc_batch.u_len[0].item()
        T = tgt_len[0].item()
        attention = u_attn_scores[0, :T, :N].sum(dim=0) / u_attn_scores[0, :T, :N].sum()
        attention = attention.cpu().numpy()
        return attention

    def get_utt_attn_without_ref(self, enc_batch, beam_width=4, time_step=144, penalty_ug=0.0, alpha=1.25, length_offset=5):
        decode_dict = {
            'k': beam_width,
            'time_step': time_step,
            'vocab_size': 30522,
            'device': self.device,
            'start_token_id': 101, 'stop_token_id': 103,
            'alpha': alpha,
            'length_offset': length_offset,
            'penalty_ug': penalty_ug,
            'keypadmask_dtype': KEYPADMASK_DTYPE,
            'memory_utt': False,
            'batch_size': 1
        }
        # batch_size should be 1
        with torch.no_grad():

            summary_ids, attn_scores, u_attn_scores = self.model.decode_beamsearch(
                    enc_batch.input, enc_batch.u_len, enc_batch.w_len, decode_dict)

        N = enc_batch.u_len[0].item()
        attention = u_attn_scores[:,:N].sum(dim=0) / u_attn_scores[:,:N].sum()
        attention = attention.cpu().numpy()
        return attention

def test():

    model = HierarchicalModel("HIERDIV", use_gpu=True)

    tokenizer = HierTokenizer()
    batches = tokenizer.get_enc_input([LONG_BORING_TENNIS_ARTICLE, LONG_BORING_TENNIS_ARTICLE], use_gpu=True)

    summaries_ug00 = model.decode(tokenizer, batches, penalty_ug=0.0)
    summaries_ug20 = model.decode(tokenizer, batches, penalty_ug=20.0)

if __name__ == "__main__":
    test()
