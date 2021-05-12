import os
import torch
import sys
import pickle
import random
from datetime import datetime
from collections import OrderedDict
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import BartTokenizer, BartForConditionalGeneration

from data.loader import BartBatcher, load_podcast_data
from data.processor import PodcastEpisode

from nltk import tokenize

from transformers.modeling_utils import BeamHypotheses, calc_banned_ngram_tokens

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def ensemble_prediction(models, input_ids, model_config, is_enc_input_same=True,
                        num_beams=4, max_length=144, min_length=56,
                        no_repeat_ngram_size=3, length_penalty=2.0):
    # models = list contain models
    # input_ids = source input ids (assuming that batch_size = 1) --- torch size [1, T]
    # if is_enc_input_same is False, input_ids is an array

    batch_size = 1
    num_return_sequences = 1

    num_models = len(models)

    # config
    early_stopping = model_config.early_stopping
    bos_token_id   = model_config.bos_token_id
    pad_token_id   = model_config.pad_token_id
    eos_token_id   = model_config.eos_token_id
    decoder_start_token_id = model_config.decoder_start_token_id

    vocab_size = model_config.vocab_size

    temperature = model_config.temperature
    use_cache = model_config.use_cache

    # for what??
    effective_batch_size = batch_size
    effective_batch_mult = 1

    # -------------------------------- ENCODER -------------------------------- #
    encoder_outputs_array = [None for _ in range(num_models)]
    if is_enc_input_same:
        assert input_ids.size(0) == 1

        if (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        else:
            # looks like only this one will be used (for batch_size = 1)
            attention_mask = input_ids.new_ones(input_ids.shape)

        for mi, model in enumerate(models):
            encoder = model.get_encoder()
            encoder_outputs: tuple = encoder(input_ids, attention_mask=attention_mask)
            encoder_outputs_array[mi] = encoder_outputs

    else:
        assert num_models == len(input_ids)
        for mi in range(num_models):
            _input_ids = input_ids[mi]
            assert _input_ids.size(0) == 1

            if (pad_token_id is not None) and (pad_token_id in _input_ids):
                _attention_mask = _input_ids.ne(pad_token_id).long()
            else:
                # looks like only this one will be used (for batch_size = 1)
                _attention_mask = _input_ids.new_ones(_input_ids.shape)

            model = models[mi]
            encoder = model.get_encoder()
            encoder_outputs: tuple = encoder(_input_ids, attention_mask=_attention_mask)
            encoder_outputs_array[mi] = encoder_outputs
    # -------------------------------- DECODER -------------------------------- #

    # if config.is_encoder_decoder
    input_ids = torch.full(
        (effective_batch_size * num_beams, 1),
        decoder_start_token_id,
        dtype=torch.long,
        device=torch_device,
    )
    cur_len = 1
    # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
    expanded_batch_idxs = (
        torch.arange(batch_size)
            .view(-1, 1)
            .repeat(1, num_beams * effective_batch_mult)
            .view(-1)
            .to(input_ids.device)
        )
    for mi in range(num_models):
        encoder_outputs_array[mi] = (encoder_outputs_array[mi][0].index_select(0, expanded_batch_idxs), *encoder_outputs_array[mi][1:])

    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
        for _ in range(batch_size)
    ]

    # scores for each sentence in the beam
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=torch_device)

    # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
    # if do_sample is False:
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

    past_array = encoder_outputs_array  # defined for encoder-decoder models, None for decoder-only models

    # done sentences
    done = [False for _ in range(batch_size)]

    while cur_len < max_length:
        model_outputs_array = [None for _ in range(num_models)]
        for mi, model in enumerate(models):
            past = past_array[mi]
            if not past[1]:
                encoder_outputs, decoder_cached_states = past, None
            else:
                encoder_outputs, decoder_cached_states = past
            model_inputs =  {
                "input_ids": None,  # encoder_outputs is defined. input_ids not needed
                "encoder_outputs": encoder_outputs,
                "decoder_cached_states": decoder_cached_states,
                "decoder_input_ids": input_ids,
                "attention_mask": None, # I think this is correct!
                "use_cache": use_cache,
            }
            model_outputs_array[mi] = model(**model_inputs)

        sum_model_outputs = torch.zeros((model_outputs_array[0][0][:,-1,:].shape), dtype=torch.float, device=torch_device)
        for mi in range(num_models):
            sum_model_outputs += model_outputs_array[mi][0][:,-1,:]
        next_token_logits = sum_model_outputs / num_models

        # if model has past, then set the past variable to speed up decoding
        for mi in range(num_models):
            if models[mi]._use_cache(model_outputs_array[mi], use_cache):
                past_array[mi] = model_outputs_array[mi][1]

        # # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        # if repetition_penalty != 1.0:
        #     self.enforce_repetition_penalty_(
        #         next_token_logits, batch_size, num_beams, input_ids, repetition_penalty,
        #     )

        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        next_token_logits = models[0].prepare_logits_for_generation(
            next_token_logits, cur_len=cur_len, max_length=max_length
        )
        scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

        if eos_token_id is not None and cur_len < min_length:
            scores[:, eos_token_id] = -float("inf")

        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            num_batch_hypotheses = batch_size * num_beams
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            banned_batch_tokens = calc_banned_ngram_tokens(
                input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                scores[i, banned_tokens] = -float("inf")

        assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
            scores.shape, (batch_size * num_beams, vocab_size)
        )
        next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

        # re-organize to group the beam together (we are keeping top hypothesis accross beams)
        next_scores = next_scores.view(
            batch_size, num_beams * vocab_size
        )  # (batch_size, num_beams * vocab_size)

        next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

        assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)
        next_batch_beam = []

        # for each sentence
        for batch_idx in range(batch_size):

            # if we are done with this sentence
            if done[batch_idx]:
                assert (
                    len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next tokens for this sentence
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                # get beam and token IDs
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size

                effective_beam_id = batch_idx * num_beams + beam_id
                # add to generated hypotheses if end of sentence or last iteration
                if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(
                        input_ids[effective_beam_id].clone(), beam_token_score.item(),
                    )
                else:
                    # add next predicted token if it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                # the beam for next step is full
                if len(next_sent_beam) == num_beams:
                    break

            # Check if were done so that we can save a pad step if all(done)
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len=cur_len
            )

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1)

        # stop when we are done with each sentence
        if all(done):
            break

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])

        # re-order batch and update current length
        input_ids = input_ids[beam_idx, :]
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
        cur_len = cur_len + 1

        # re-order internal states
        for mi in range(num_models):
            if past_array[mi] is not None:
                past_array[mi] = models[mi]._reorder_cache(past_array[mi], beam_idx)

    # finalize all open beam hypotheses and end to generated hypotheses
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue

        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        if eos_token_id is not None and all(
            (token_id % vocab_size).item() is not eos_token_id for token_id in next_tokens[batch_idx]
        ):
            assert torch.all(
                next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
            ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
            )

        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)

    # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
    output_batch_size = batch_size * num_return_sequences
    output_num_return_sequences_per_batch = num_return_sequences

    # select the best hypotheses
    sent_lengths = input_ids.new(output_batch_size)
    best = []

    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)

    # shorter batches are filled with pad_token
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`Pad_token_id` has to be defined"
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

        # fill with hypothesis and eos_token_id if necessary
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
    else:
        # none of the hypotheses have an eos_token
        assert (len(hypo) == max_length for hypo in best)
        decoded = torch.stack(best).type(torch.long).to(torch_device)

    return decoded

def load_bartvanilla_model(path, load_option=0):
    bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    if torch_device == 'cuda':
        bart.cuda()
        state = torch.load(path)
    else:
        state = torch.load(path, map_location=torch.device('cpu'))

    model_state_dict = state['model']
    if load_option == 0:
        new_model_state_dict = OrderedDict()
        for key in model_state_dict.keys():
            if "module.bart." in key:
                new_model_state_dict[key.replace("module.bart.","")] = model_state_dict[key]
            elif "module." in key:
                new_model_state_dict[key.replace("module.","")] = model_state_dict[key]
            else:
                print("key {} found".format(key))
                import pdb; pdb.set_trace()
        bart.load_state_dict(new_model_state_dict)
    else:
        bart.load_state_dict(model_state_dict)

    del state, model_state_dict
    torch.cuda.empty_cache()

    return bart

DATA_PATH_HIE1040   = "/home/alta/summary/pm574/podcast_sum0/lib/test_data/filtered_hier30k_1040/podcast_testset.bin"
ENSEMBLE_NAME = "AUG30-v1"
DECODE_DIR  = "/home/alta/summary/pm574/podcast_sum0/work_submission/system_output/ensemble/{}/".format(ENSEMBLE_NAME)

def decode(start_id, end_id):
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    MODEL_PATH = "/home/alta/summary/pm574/podcast_sum0/lib/trained_models/{}.pt"
    model1_path  = MODEL_PATH.format("bartvanilla-RL-xsum-AUG12-seed2001-step30000.red")
    model2_path  = MODEL_PATH.format("bartvanilla-RL-xsum-AUG12-seed2002-step30000.red")
    model3_path  = MODEL_PATH.format("bartvanilla-RL-xsum-AUG12-seed2003-step30000.red")

    model1 = load_bartvanilla_model(model1_path, load_option=1)
    model2 = load_bartvanilla_model(model2_path, load_option=1)
    model3 = load_bartvanilla_model(model3_path, load_option=1)

    ensemble = [model1, model2, model3]

    with open(DATA_PATH_HIE1040, 'rb') as f:
        podcasts1 = pickle.load(f, encoding="bytes")
    print("len(podcasts1) = {}".format(len(podcasts1)))


    ids = [x for x in range(start_id, end_id)]
    random.shuffle(ids)

    for id in ids:
        # check if the file exist or not
        out_path = "{}/{}_decoded.txt".format(DECODE_DIR, id)
        exist = os.path.isfile(out_path)
        if exist:
            print("id {}: already exists".format(id))
            continue

        article_input_ids1 = bart_tokenizer.batch_encode_plus([podcasts1[id].transcription],
                            return_tensors='pt', max_length=model2.config.max_position_embeddings)['input_ids'].to(torch_device)

        with torch.no_grad():
            summary_ids = ensemble_prediction(ensemble,
                            [article_input_ids1, article_input_ids1, article_input_ids1],
                            model1.config,
                            is_enc_input_same=False,
                            num_beams=4,
                            length_penalty=2.0,
                            max_length=144, # set this equal to the max length in training
                            min_length=56,  # two sentences
                            no_repeat_ngram_size=3
                        )

        summary_txt = bart_tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

        # [SEP] tokens were not added explicitly between sentences in training, so need to use tokenizer
        generated_sentences = tokenize.sent_tokenize(summary_txt.strip())
        with open(out_path, 'w') as file:
            file.write("\n".join(generated_sentences))
        print("write:", out_path)


if __name__ == "__main__":
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
        print("Usage: python ensemble_decode_testset.py start_id end_id")
        raise Exception("argv error")
