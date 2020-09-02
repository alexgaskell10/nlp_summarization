import itertools
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List
import sys

import git
import numpy as np
import torch
from rouge_score import rouge_scorer, scoring
from torch import nn
from torch.utils.data import Dataset, Sampler
import torch.nn.functional as F
from tqdm import tqdm
from .tokenization_bart_custom import CustomBartTokenizer, CustomTruncationStrategy


def encode_file(
    tokenizer,
    data_path,
    max_length,
    pad_to_max_length=True,
    return_tensors="pt",
    overwrite_cache=False,
    prefix="",
    tok_name="",
    truncation_strategy=None,
):
    cache_path = Path(f"{data_path}_{tok_name}{max_length}.pt")
    if not overwrite_cache and cache_path.exists():
        try:
            examples = torch.load(str(cache_path))
            assert isinstance(examples, list)
            return examples

        except Exception:
            print(f"failed to load from {cache_path}, retokenizing {data_path}")
    data_path = Path(data_path)

    lns = lmap(str.strip, data_path.open().readlines())
    lns = [prefix + text for text in lns]
    assert lns, f"found empty file at {data_path}"
    examples = []
    for text in tqdm(lns, desc=f"Tokenizing {data_path.name}"):
        tokenized = tokenizer.batch_encode_plus(
            [text],  # DONT ADD SPACES
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
            add_prefix_space=True,
            truncation=True,
            return_tensors=return_tensors,
            truncation_strategy=truncation_strategy,
        )
        assert tokenized.input_ids.shape[1] == max_length
        examples.append(tokenized)

    if truncation_strategy == CustomTruncationStrategy.RANDOM_START:
        outfile = Path(f"{data_path}_{tok_name}{max_length}_randomstartinfo.txt")
        tokenizer.save_random_start_info(outfile)

    torch.save(lmap(dict, examples), cache_path.open("wb"))

    return examples


def lmap(f, x):
    return list(map(f, x))


T5_PREFIX = "summarize: "  # HACK, fixme


def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class SummarizationDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        type_path="train",
        max_source_length=1024,
        max_target_length=56,
        n_obs=None,
        overwrite_cache=False,
        prefix="",
        pad_to_length=False,
    ):
        super().__init__()
        tok_name = tokenizer.__class__.__name__.lower().rstrip("tokenizer")
        
        truncation_strategy = CustomTruncationStrategy.RANDOM_START if isinstance(tokenizer, CustomBartTokenizer) else None

        self.source = encode_file(
            tokenizer,
            os.path.join(data_dir, type_path + ".source"),
            max_source_length,
            overwrite_cache=overwrite_cache,
            prefix=prefix,
            tok_name=tok_name,
            truncation_strategy=truncation_strategy,
        )
        tgt_path = os.path.join(data_dir, type_path + ".target")
        self.target = encode_file(
            tokenizer, tgt_path, max_target_length, overwrite_cache=overwrite_cache, tok_name=tok_name
        )
        if n_obs is not None:
            self.source = self.source[:n_obs]
            self.target = self.target[:n_obs]
        self.pad_token_id = tokenizer.pad_token_id if not pad_to_length else -100

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()
        return {"input_ids": source_ids, "attention_mask": src_mask, "decoder_input_ids": target_ids}

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        y = trim_batch(batch["decoder_input_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["input_ids"], pad_token_id, attention_mask=batch["attention_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch) -> dict:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["decoder_input_ids"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        batch = {"input_ids": source_ids, "attention_mask": source_mask, "decoder_input_ids": y}
        return batch

    @property
    def src_lens(self):  # Can delete?
        return lmap(len, self.source)

    @property
    def tgt_lens(self):
        return lmap(len, self.target)

    def make_sortish_sampler(self, batch_size):
        return SortishSampler(self.source, batch_size)


class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size):
        self.data, self.bs = data, batch_size

    def key(self, i):
        return len(self.data[i])

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        idxs = np.random.permutation(len(self.data))
        sz = self.bs * 50
        ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([sorted(s, key=self.key, reverse=True) for s in ck_idx])
        sz = self.bs
        ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
        ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)


def use_task_specific_params(model, task):
    # update config with summarization specific params
    task_specific_params = model.config.task_specific_params
    if task_specific_params is not None:
        model.config.update(task_specific_params.get(task, {}))


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


def save_git_info(folder_path: str):
    """
    Log commit info.
    """
    repo_infos = get_git_info()

    with open(os.path.join(folder_path, "git_log.json"), "w") as f:
        json.dump(repo_infos, f, indent=4)


def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    repo_infos = {
        "repo_id": str(repo),
        "repo_sha": str(repo.head.object.hexsha),
        "repo_branch": str(repo.active_branch),
    }
    return repo_infos


ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]


def calculate_rouge(output_lns: List[str], reference_lns: List[str]) -> Dict:
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: v.mid.fmeasure for k, v in result.items()}


def freeze_params(model: nn.Module):
    for par in model.parameters():
        par.requires_grad = False


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        bs = pad_mask.long().sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        bs = lprobs.shape[0]

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss / bs, nll_loss / bs


def random_encoder_attn_weights(model):
    ''' Randomises the weights of the self attention layers of a model
        (encoder only)

        return:
        - model: BartForConditionalGeneration
    '''
    from torch.nn.init import kaiming_normal_ as kaiming_normal

    for layer in model.model.encoder.layers:
        layer.self_attn.longformer_self_attn.key.weight = kaiming_normal(layer.self_attn.longformer_self_attn.key.weight)
        layer.self_attn.longformer_self_attn.value.weight = kaiming_normal(layer.self_attn.longformer_self_attn.value.weight)
        layer.self_attn.longformer_self_attn.query.weight = kaiming_normal(layer.self_attn.longformer_self_attn.query.weight)

    return model


def get_led(args):
    ''' Helper to create the longformer encoder decoder model
        returns:
        - model: BartForConditionalGeneration
        - tokenizer: BartTokenizer
    '''
    sys.path.append('./led')
    from led.longbart.convert_bart_to_longbart import create_long_model

    model_path = "led"

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # model_name_or_path can be a name of model (e.g. facebook/bart-large) 
    # or a path to a dir containing a model to be loaded
    base_model = 'facebook/bart-large-cnn' if 'best_tfmr' in args.model_name_or_path else args.model_name_or_path

    model, tokenizer = create_long_model(
        save_model_to=model_path,
        base_model=base_model,
        attention_window=args.attn_window,
        max_pos=4096,
    )

    if args.do_train and hasattr(args, 'random_weights') and args.random_weights:
        from hf.utils import random_encoder_attn_weights
        model = random_encoder_attn_weights(model)

    return model, tokenizer


def get_red(args):
    ''' Helper to create the reformer encoder decoder model
        returns:
        - model: BartForConditionalGeneration
        - tokenizer: BartTokenizer
    '''
    sys.path.append('./red')
    from red.convert_bart_to_reformerencoderdecoder import create_long_model

    base_model = args.longbart_base_model
    model_path = "red"

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model, tokenizer = create_long_model(
        save_model_to=None,
        base_model=args.longbart_base_model,
        attn_type=args.reformerencoderdecoder_attn_type,
        max_pos=args.max_source_length,
    )

    return model, tokenizer


def get_bart(args):
    ''' Helper to return a bart model (used if we only want to run eval)
        returns:
        - model: BartForConditionalGeneration
        - tokenizer: BartTokenizer
    '''
    sys.path.append('./huggingface')
    from huggingface.modeling_bart import BartForConditionalGeneration
    from transformers import BartTokenizer

    model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
    tokenizer = BartTokenizer.from_pretrained(
        args.model_name_or_path, 
        model_max_length=args.max_source_length
    )

    return model, tokenizer
