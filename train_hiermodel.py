import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import random
from datetime import datetime
from collections import OrderedDict

from transformers import BertTokenizer
from hierarchical_rnn_v5 import EncoderDecoder
from nltk import tokenize
from data.loader import BartBatcher, load_podcast_data
from data.processor import PodcastEpisode

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

class HierBatcher(BartBatcher):
    def __init__(self, tokenizer, config, podcasts, torch_device):
        super().__init__(tokenizer, config, podcasts, torch_device)

        self.num_utterances = config['num_utterances']
        self.num_words      = config['num_words']
        self.summary_length = config['summary_length']

    # Override
    def get_a_batch(self, batch_size):
        """
        return input, u_len, w_len, target, tgt_len
        """

        input = np.zeros((batch_size, self.num_utterances, self.num_words), dtype=np.long)
        u_len = np.zeros((batch_size), dtype=np.long)
        w_len = np.zeros((batch_size, self.num_utterances), dtype=np.long)

        target  = np.zeros((batch_size, self.summary_length), dtype=np.long)
        target.fill(103)
        tgt_len = np.zeros((batch_size), dtype=np.int)

        batch_count = 0
        while batch_count < batch_size:
            if not self.is_podcast_good(self.cur_id):
                self.increment_pod_id()
                continue

            # inputs[batch_count]  = self.podcasts[self.cur_id].transcription
            # targets[batch_count] = self.podcasts[self.cur_id].description
            transcription = self.podcasts[self.cur_id].transcription.lower()
            description   = self.podcasts[self.cur_id].description.lower()

            # ENCODER
            sentences = tokenize.sent_tokenize(transcription)
            num_sentences = len(sentences)
            if num_sentences > self.num_utterances:
                num_sentences = self.num_utterances
                sentences = sentences[:self.num_utterances]
            u_len[batch_count] = num_sentences

            for j, sent in enumerate(sentences):
                token_ids = self.tokenizer.encode(sent)[1:-1] # remove [CLS], [SEP]
                utt_len = len(token_ids)
                if utt_len > self.num_words:
                    utt_len = self.num_words
                    token_ids = token_ids[:self.num_words]
                input[batch_count,j,:utt_len] = token_ids
                w_len[batch_count,j] = utt_len

            # DECODER
            concat_tokens = [101]
            sentences = tokenize.sent_tokenize(description)
            for j, sent in enumerate(sentences):
                token_ids = self.tokenizer.encode(sent)[1:-1] # remove [CLS], [SEP]
                concat_tokens.extend(token_ids)
                concat_tokens.extend([102]) # [SEP]
            tl = len(concat_tokens)
            if tl > self.summary_length:
                concat_tokens = concat_tokens[:self.summary_length]
                tl = self.summary_length
            target[batch_count, :tl] = concat_tokens
            tgt_len[batch_count] = tl

            # increment
            self.increment_pod_id()
            batch_count += 1

        input = torch.from_numpy(input).to(self.device)
        u_len = torch.from_numpy(u_len).to(self.device)
        w_len = torch.from_numpy(w_len).to(self.device)
        target = torch.from_numpy(target).to(self.device)

        return input, u_len, w_len, target, tgt_len

def train():
    print("Start training hierarchical RNN model")
    # ---------------------------------------------------------------------------------- #
    args = {}
    args['use_gpu']        = True
    args['num_utterances'] = 360   # max no. utterance in a meeting
    args['num_words']      = 40    # max no. words in an utterance
    args['summary_length'] = 144   # max no. words in a summary
    args['vocab_size']     = 30522 # BERT tokenizer
    args['embedding_dim']   = 256   # word embeeding dimension
    args['rnn_hidden_size'] = 512 # RNN hidden size

    args['dropout']        = 0.1
    args['num_layers_enc'] = 2    # in total it's num_layers_enc*2 (word/utt)
    args['num_layers_dec'] = 1

    args['random_seed'] = 78
    args['memory_utt']  = False

    args['model_save_dir'] = "/home/alta/summary/pm574/summariser1/lib/trained_models_spotify/"
    # args['load_model'] = "/home/alta/summary/pm574/summariser1/lib/trained_models2/model-HGRUV5_CNNDMDIV_APR14A-ep02.pt"
    args['model_name'] = 'HGRUV5DIV_SPOTIFY_JUNE18_v4'
    # ---------------------------------------------------------------------------------- #
    print_config(args)

    if args['use_gpu']:
        if 'X_SGE_CUDA_DEVICE' in os.environ: # to run on CUED stack machine
            print('running on the stack... 1 GPU')
            cuda_device = os.environ['X_SGE_CUDA_DEVICE']
            print('X_SGE_CUDA_DEVICE is set to {}'.format(cuda_device))
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
        else:
            print('running locally...')
            os.environ["CUDA_VISIBLE_DEVICES"] = '0,1' # choose the device (GPU) here
        device = 'cuda'
    else:
        device = 'cpu'
    print("device = {}".format(device))

    # random seed
    random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    np.random.seed(args['random_seed'])

    # Data
    podcasts = load_podcast_data(sets=-1)
    batcher = HierBatcher(bert_tokenizer, args, podcasts, device)
    val_podcasts = load_podcast_data(sets=[10])
    val_batcher = HierBatcher(bert_tokenizer, args, val_podcasts, device)

    model = EncoderDecoder(args, device=device)
    print(model)

    # Load model if specified (path to pytorch .pt)
    # state = torch.load(args['load_model'])
    # model_state_dict = state['model']
    # model.load_state_dict(model_state_dict)
    # print("load succesful #1")

    criterion = nn.NLLLoss(reduction='none')

    # we use two separate optimisers (encoder & decoder)
    optimizer = optim.Adam(model.parameters(),lr=2e-20,betas=(0.9,0.999),eps=1e-08,weight_decay=0)
    optimizer.zero_grad()

    # validation losses
    training_step  = 0
    batch_size     = 4
    gradient_accum = 1
    total_step     = 1000000
    valid_step     = 2000
    best_val_loss  = 99999999

    # to use multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Multiple GPUs: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    while training_step < total_step:

            # get a batch
            input, u_len, w_len, target, tgt_len = batcher.get_a_batch(batch_size)

            # decoder target
            decoder_target, decoder_mask = shift_decoder_target(target, tgt_len, device, mask_offset=True)
            decoder_target = decoder_target.view(-1)
            decoder_mask = decoder_mask.view(-1)

            decoder_output, _, _, _, u_attn_scores = model(input, u_len, w_len, target)
            loss = criterion(decoder_output.view(-1, args['vocab_size']), decoder_target)
            loss = (loss * decoder_mask).sum() / decoder_mask.sum()
            loss.backward()


            if training_step % gradient_accum == 0:
                adjust_lr(optimizer, training_step)
                optimizer.step()
                optimizer.zero_grad()

            if training_step % 1 == 0:
                print("[{}] step {}/{}: loss = {:.5f}".format(
                    str(datetime.now()), training_step, total_step, loss))
                sys.stdout.flush()

            # if training_step % 10 == 0:
            #     print("======================== GENERATED SUMMARY ========================")
            #     print(bert_tokenizer.decode(torch.argmax(decoder_output[0], dim=-1).cpu().numpy()[:tgt_len[0]]))
            #     print("======================== REFERENCE SUMMARY ========================")
            #     print(bert_tokenizer.decode(decoder_target.view(batch_size,args['summary_length'])[0,:tgt_len[0]].cpu().numpy()))

            if training_step % valid_step == 0 and training_step > 5:
                # ---------------- Evaluate the model on validation data ---------------- #
                print("Evaluating the model at training step {}".format(training_step))
                print("learning_rate = {}".format(optimizer.param_groups[0]['lr']))
                # switch to evaluation mode
                model.eval()
                with torch.no_grad():
                    valid_loss = evaluate(model, val_batcher, batch_size, args, device)
                print("valid_loss = {}".format(valid_loss))
                # switch to training mode
                model.train()
                if valid_loss < best_val_loss:
                    stop_counter = 0
                    best_val_loss = valid_loss
                    print("Model improved".format(stop_counter))
                else:
                    stop_counter += 1
                    print("Model not improved #{}".format(stop_counter))
                    if stop_counter == 3:
                        print("Stop training!")
                        return
                state = {
                    'training_step': training_step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                }
                savepath = args['model_save_dir']+"{}-step{}.pt".format(args['model_name'],training_step)
                torch.save(state, savepath)
                print("Saved at {}".format(savepath))

            training_step += 1

    print("End of training hierarchical RNN model")

def evaluate(model, val_batcher, batch_size, args, device):
    print("start validating")
    criterion = nn.NLLLoss(reduction='none')
    eval_total_loss = 0.0
    eval_total_tokens = 0
    while val_batcher.epoch_counter < 1:
    # for i in range(5):
        input, u_len, w_len, target, tgt_len = val_batcher.get_a_batch(batch_size)
        # decoder target
        decoder_target, decoder_mask = shift_decoder_target(target, tgt_len, device)
        decoder_target = decoder_target.view(-1)
        decoder_mask = decoder_mask.view(-1)
        decoder_output = model(input, u_len, w_len, target)
        # decoder_output, _, _, _, _ = model(input, u_len, w_len, target)
        loss = criterion(decoder_output.view(-1, args['vocab_size']), decoder_target)
        eval_total_loss   += (loss * decoder_mask).sum().item()
        eval_total_tokens += decoder_mask.sum().item()
        print("#", end="")
        sys.stdout.flush()
    print()
    avg_eval_loss = eval_total_loss / eval_total_tokens
    val_batcher.epoch_counter = 0
    val_batcher.cur_id = 0
    print("finish validating")
    return avg_eval_loss

def adjust_lr(optimizer, step, warmup=2000):
    """to adjust the learning rate"""
    step = step + 1 # plus 1 to avoid ZeroDivisionError
    lr = 2e-2 * min(step**(-0.5), step*(warmup**(-1.5)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return

def shift_decoder_target(target, tgt_len, device, mask_offset=False):
    # MASK_TOKEN_ID = 103
    batch_size = target.size(0)
    max_len = target.size(1)
    dtype0  = target.dtype

    decoder_target = torch.zeros((batch_size, max_len), dtype=dtype0, device=device)
    decoder_target[:,:-1] = target.clone().detach()[:,1:]
    # decoder_target[:,-1:] = 103 # MASK_TOKEN_ID = 103
    # decoder_target[:,-1:] = 0 # add padding id instead of MASK

    # mask for shifted decoder target
    decoder_mask = torch.zeros((batch_size, max_len), dtype=torch.float, device=device)
    if mask_offset:
        offset = 10
        for bn, l in enumerate(tgt_len):
            # decoder_mask[bn,:l-1].fill_(1.0)
            # to accommodate like 10 more [MASK] [MASK] [MASK] [MASK],...
            if l-1+offset < max_len: decoder_mask[bn,:l-1+offset].fill_(1.0)
            else: decoder_mask[bn,:].fill_(1.0)
    else:
        for bn, l in enumerate(tgt_len):
            decoder_mask[bn,:l-1].fill_(1.0)

    return decoder_target, decoder_mask

def print_config(args):
    print("============================= CONFIGURATION =============================")
    for x in args:
        print('{}={}'.format(x, args[x]))
    print("=========================================================================")

if __name__ == "__main__":
    # ------ TRAINING ------ #
    train()
