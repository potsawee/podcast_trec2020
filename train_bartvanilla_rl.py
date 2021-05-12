import os
import sys
import random
from datetime import datetime
from collections import OrderedDict
from rouge_score import rouge_scorer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_bart import fill_with_neg_inf
from transformers.modeling_utils import top_k_top_p_filtering

from data.loader import BartBatcher, load_podcast_data, load_podcast_hier30k_1040_filtered_data
from data.processor import PodcastEpisode

from ensemble_decode_testset import load_bartvanilla_model

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

SAVE_DIR   = "/home/alta/summary/pm574/podcast_sum0/lib/trained_models"
MODEL_NAME = "bartvanilla-podcast-RL"

def train():
    # Model & Optimizer
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # Option1: Train from BART-cnndm, BART-xsum
    # bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    # Option2 : Train from BART-podcast
    MODEL_PATH = "/home/alta/summary/pm574/podcast_sum0/lib/trained_models/{}.pt"
    model1_path  = MODEL_PATH.format("bartvanilla-hier30k1040-xsum-JULY28-v1-step180000")
    bart = load_bartvanilla_model(model1_path, load_option=1)

    # Freeze some layers --- should instead  use larger GPU, e.g. 16GB or use apex fp16 bit training!!!
    # In the TREC2020, I froze layers as the code shown below ---> this definitely leads to some degradation
    num_freeze_layers = 3
    print("num_freeze_layers:", num_freeze_layers)
    for _k in range(num_freeze_layers):
        for param in bart.model.encoder.layers[_k].parameters(): param.requires_grad = False
    for _k in range(num_freeze_layers):
        for param in bart.model.decoder.layers[_k].parameters(): param.requires_grad = False

    bart.train()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, bart.parameters()), lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=0)
    optimizer.zero_grad()

    bart_config = bart.model.config
    print(bart)
    print(bart_config)
    if torch_device == 'cuda': bart.cuda()
    print("#parameters:", sum(p.numel() for p in bart.parameters() if p.requires_grad))

    # Data
    podcasts = load_podcast_hier30k_1040_filtered_data(sets=-1) # -1 means set0,..,set9 (excluding 10)
    batcher = BartBatcher(bart_tokenizer, bart.model.config, podcasts, torch_device)

    # Validation
    val_podcasts = load_podcast_hier30k_1040_filtered_data(sets=[10])
    val_batcher = BartBatcher(bart_tokenizer, bart.model.config, val_podcasts, torch_device)

    # Criterion
    criterion = nn.CrossEntropyLoss(reduction='none') # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

    training_step  = 0
    batch_size     = 1
    gradient_accum = 2
    valid_step     = 2000
    total_step     = 2000 * 1000
    best_val_loss  = 99999999
    random_seed    = 2004
    stop_counter   = 0

    min_decode_len = 20
    max_decode_len = 64
    gamma          = 0.9
    max_norm       = 50.0


    print("batch_size:", batch_size)
    print("training_step:", training_step)
    print("gradient_accum:", gradient_accum)
    print("total_step:", total_step)
    print("valid_step:", valid_step)
    print("random_seed:", random_seed)
    print("min_decode_len:", min_decode_len)
    print("max_decode_len:", max_decode_len)
    print("gamma:", gamma)
    print("max_norm:", max_norm)


    # Randomness
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # shuffle data
    batcher.shuffle_podcasts()

    rouge_sc = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

    decoder_input_ids = torch.tensor([[bart.config.decoder_start_token_id] for _ in range(batch_size)], device=torch_device)

    while training_step < total_step:
        # get a batch
        input_ids, attention_mask, target_ids, target_attention_mask = batcher.get_a_batch(batch_size=batch_size)
        shifted_target_ids, shifted_target_attention_mask = batcher.shifted_target_left(target_ids, target_attention_mask)

        # Encoder!
        encoder_outputs = bart.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
        )

        assert isinstance(encoder_outputs, tuple)

        # Decoder - Teacher Forcing
        # BART forward
        if gamma < 1.0:
            tf_x = bart(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=target_ids,
                decoder_attention_mask=target_attention_mask,
            )
            # x[0] # decoder output
            # x[1] # encoder output
            tf_lm_logits = tf_x[0]

            loss_ml = criterion(tf_lm_logits.view(-1, bart_config.vocab_size), shifted_target_ids.view(-1))
            shifted_target_attention_mask = shifted_target_attention_mask.view(-1)
            loss_ml = (loss_ml * shifted_target_attention_mask).sum() / shifted_target_attention_mask.sum()
        else:
            loss_ml = 0

        if gamma > 0.0:
            # Sampling
            generated_tokens, output_probs = this_generate_no_beam_search(
                        bart, input_ids=decoder_input_ids,
                        encoder_outputs=encoder_outputs, attention_mask=attention_mask,
                        min_length=min_decode_len, max_length=max_decode_len, do_sample=True,
                        temperature=1.0, top_k=50, top_p=1.0,
                        batch_size=batch_size, use_cache=True)

            assert batch_size == 1 # only support batch_size = 1 for now
            seq_len = generated_tokens.size(1)
            generated_tokens = generated_tokens[0].cpu()
            output_probs     = output_probs[0]

            total_log_prob = 0
            for t in range(seq_len):
                tok  = generated_tokens[t]
                prob = output_probs[t, tok]
                total_log_prob += torch.log(prob)

            # Greedy Search
            with torch.no_grad():
                generated_tokens_gd, _ = this_generate_no_beam_search(
                            bart, input_ids=decoder_input_ids,
                            encoder_outputs=encoder_outputs, attention_mask=attention_mask,
                            min_length=min_decode_len, max_length=max_decode_len, do_sample=False,
                            temperature=None, top_k=None, top_p=None,
                            batch_size=batch_size, use_cache=True)

            sample_seq = bart_tokenizer.decode(generated_tokens.squeeze(), skip_special_tokens=True).strip()
            argmax_seq = bart_tokenizer.decode(generated_tokens_gd.squeeze(), skip_special_tokens=True).strip()
            target_seq = bart_tokenizer.decode(target_ids.cpu().squeeze(), skip_special_tokens=True).strip()

            sample_scores = rouge_sc.score(sample_seq, target_seq)
            argmax_scores = rouge_sc.score(argmax_seq, target_seq)

            sample_rougeL = sample_scores['rougeLsum'].fmeasure
            argmax_rougeL = argmax_scores['rougeLsum'].fmeasure

            reward = sample_rougeL - argmax_rougeL
            loss_rl = - reward * total_log_prob

        else:
            loss_rl = 0.0

        loss = gamma*loss_rl + (1-gamma)*loss_ml
        loss.backward()

        if (training_step+1) % gradient_accum == 0:
            # gradient clipping (norm)
            nn.utils.clip_grad_norm_(bart.parameters(), max_norm)
            # total_norm = 0
            # for paaa in bart.parameters():
            #     if paaa.requires_grad:
            #         param_norm = paaa.grad.data.norm(2)
            #         total_norm += param_norm.item() ** 2
            #     else:
            #         pass
            # total_norm = total_norm ** (1. / 2)
            # print("total_norm:", total_norm)

            adjust_lr(optimizer, training_step)
            optimizer.step()
            optimizer.zero_grad()

        if training_step % 10 == 0:
            print("[{}] step {}/{}: loss_ml = {:.5f} | loss_rl = {:.5f}".format(
                str(datetime.now()), training_step, total_step, loss_ml, loss_rl))
            sys.stdout.flush()

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
    lr = 0.1 * 2e-3 * min(step**(-0.5), step*(warmup**(-1.5)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return

def validation(bart, bart_config, val_podcasts, val_batcher, batch_size):
    print("start validating")
    criterion = nn.CrossEntropyLoss(reduction='none')
    sum_loss = 0
    sum_token = 0
    while val_batcher.epoch_counter < 1:
    # for i in range(5):
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


def this_generate_no_beam_search(
    bart, input_ids, encoder_outputs, attention_mask,
    min_length, max_length, do_sample,
    temperature, top_k, top_p, batch_size, use_cache,
):
    """ Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
    """
    # HARD code
    cur_len = 1
    bos_token_id = bart.config.bos_token_id
    pad_token_id = bart.config.pad_token_id
    eos_token_id = bart.config.eos_token_id
    decoder_start_token_id = bart.config.decoder_start_token_id


    # length of generated sentences / unfinished sentences
    unfinished_sents = input_ids.new(batch_size).fill_(1)
    sent_lengths = input_ids.new(batch_size).fill_(max_length)

    past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models

    output_probs = None

    while cur_len < max_length:
        model_inputs = bart.prepare_inputs_for_generation(
            input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache
        )

        outputs = bart(**model_inputs)
        next_token_logits = outputs[0][:, -1, :]

        # if model has past, then set the past variable to speed up decoding
        if bart._use_cache(outputs, use_cache):
            past = outputs[1]

        # set eos token prob to zero if min_length is not reached
        if eos_token_id is not None and cur_len < min_length:
            next_token_logits[:, eos_token_id] = -float("inf")

        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            # Top-p/top-k filtering
            next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            if output_probs is None:
                output_probs = probs.unsqueeze(1)
            else:
                output_probs = torch.cat([output_probs, probs.unsqueeze(1)], dim=1)
        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)

        # update generations and finished sentences
        if eos_token_id is not None:
            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
        else:
            tokens_to_add = next_token

        # add token and increase length by one
        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        cur_len = cur_len + 1

        if eos_token_id is not None:
            eos_in_sents = tokens_to_add == eos_token_id
            # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
            is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
            sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
            # unfinished_sents is set to zero if eos in sentence
            unfinished_sents.mul_((~eos_in_sents).long())

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break

        # extend attention_mask for new generated input if only decoder
        if bart.config.is_encoder_decoder is False:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

    # if there are different sentences lengths in the batch, some batches have to be padded
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
        # finished sents are filled with pad_token
        decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)
    else:
        decoded = input_ids

    for hypo_idx, hypo in enumerate(input_ids):
        decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

    return decoded[:, 1:].cpu(), output_probs

if __name__ == "__main__":
    train()
