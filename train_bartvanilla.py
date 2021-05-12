import os
import sys
import random
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BartTokenizer, BartForConditionalGeneration

from data.loader import BartBatcher, load_podcast_data
from data.processor import PodcastEpisode

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

SAVE_DIR   = "/home/alta/summary/pm574/podcast_sum0/lib/trained_models"
MODEL_NAME = "bartvanilla-podcast-X"

def train():
    # Model & Optimizer
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')
    bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-xsum')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, bart.parameters()), lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=0)
    optimizer.zero_grad()

    bart_config = bart.model.config
    print(bart)
    print(bart_config)
    if torch_device == 'cuda': bart.cuda()
    print("#parameters:", sum(p.numel() for p in bart.parameters() if p.requires_grad))
    bart.train()

    # Data
    podcasts = load_podcast_data(sets=-1) # -1 means set0,..,set9 (excluding 10)
    batcher = BartBatcher(bart_tokenizer, bart.model.config, podcasts, torch_device)

    # Validation
    val_podcasts = load_podcast_data(sets=[10])
    val_batcher = BartBatcher(bart_tokenizer, bart.model.config, val_podcasts, torch_device)

    # Criterion
    criterion = nn.CrossEntropyLoss(reduction='none') # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

    training_step  = 0
    batch_size     = 1
    gradient_accum = 2
    valid_step     = 20000 # every a few hours on lapaz machine (1GPU - 1080Ti)
    total_step     = 20000 * 1000
    best_val_loss  = 99999999
    random_seed    = 777
    stop_counter   = 0

    print("batch_size:", batch_size)
    print("training_step:", training_step)
    print("gradient_accum:", gradient_accum)
    print("total_step:", total_step)
    print("valid_step:", valid_step)
    print("random_seed:", random_seed)

    # Randomness
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # shuffle data
    batcher.shuffle_podcasts()

    if torch.cuda.device_count() > 1:
        print("Multiple GPUs: {}".format(torch.cuda.device_count()))
        bart = nn.DataParallel(bart)

    while training_step < total_step:
        # get a batch
        input_ids, attention_mask, target_ids, target_attention_mask = batcher.get_a_batch(batch_size=batch_size)
        shifted_target_ids, shifted_target_attention_mask = batcher.shifted_target_left(target_ids, target_attention_mask)
        # BART forward
        x = bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=target_ids,
            decoder_attention_mask=target_attention_mask,
        )
        # x[0] # decoder output
        # x[1] # encoder output
        lm_logits = x[0]

        loss = criterion(lm_logits.view(-1, bart_config.vocab_size), shifted_target_ids.view(-1))
        shifted_target_attention_mask = shifted_target_attention_mask.view(-1)
        loss = (loss * shifted_target_attention_mask).sum() / shifted_target_attention_mask.sum()
        loss.backward()

        if training_step % gradient_accum == 0:
            adjust_lr(optimizer, training_step)
            optimizer.step()
            optimizer.zero_grad()

        if training_step % 1 == 0:
            print("[{}] step {}/{}: loss = {:.5f}".format(str(datetime.now()), training_step, total_step, loss))
            sys.stdout.flush()

        # if training_step % 5 == 0:
        #     tgt_len = target_attention_mask[0].sum().item()
        #     print("REF: {}".format(bart_tokenizer.decode(shifted_target_ids[0,:tgt_len].cpu().numpy())))
        #     print("HYP: {}".format(bart_tokenizer.decode(torch.argmax(lm_logits[0,:tgt_len].cpu(), dim=-1).numpy())))

        if training_step % valid_step == 0 and training_step > 5:
            bart.eval()
            with torch.no_grad():
                valid_loss = validation(bart, bart_config, val_podcasts, val_batcher, batch_size)
            print("Valid Loss = {:.5f}".format(valid_loss))
            bart.train()
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
                'model': bart.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
            savepath = "{}/{}-step{}.pt".format(SAVE_DIR, MODEL_NAME, training_step)
            torch.save(state, savepath)
            print("Saved at {}".format(savepath))

        training_step += 1
    print("Finish Training")

def adjust_lr(optimizer, step, warmup=10000):
    """to adjust the learning rate"""
    step = step + 1 # plus 1 to avoid ZeroDivisionError
    lr = 2e-3 * min(step**(-0.5), step*(warmup**(-1.5))) # 0.5 for effecetive batch_size
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return

def validation(bart, bart_config, val_podcasts, val_batcher, batch_size):
    print("start validating")
    criterion = nn.CrossEntropyLoss(reduction='none')
    sum_loss = 0
    sum_token = 0
    while val_batcher.epoch_counter < 1:
        input_ids, attention_mask, target_ids, target_attention_mask = val_batcher.get_a_batch(batch_size=batch_size)
        shifted_target_ids, shifted_target_attention_mask = val_batcher.shifted_target_left(target_ids, target_attention_mask)
        x = bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=target_ids,
            decoder_attention_mask=target_attention_mask,
        )
        lm_logits = x[0]
        loss = criterion(lm_logits.view(-1, bart_config.vocab_size), shifted_target_ids.view(-1))
        shifted_target_attention_mask = shifted_target_attention_mask.view(-1)
        sum_loss += (loss * shifted_target_attention_mask).sum().item()
        sum_token += shifted_target_attention_mask.sum().item()
        print("#", end="")
        sys.stdout.flush()
    print()
    val_batcher.epoch_counter = 0
    val_batcher.cur_id = 0
    print("finish validating")

    return sum_loss / sum_token

if __name__ == "__main__":
    train()
