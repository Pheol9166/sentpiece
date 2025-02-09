from typing import Dict
from model_pb2 import ModelProto
import json


def load_json(path: str):
    with open(path, "r") as json_file:
        return json.load(json_file)


def load_model(path: str):
    model_proto = ModelProto()
    with open(path, "rb") as fr:
        model_proto.ParseFromString(fr.read())

    vocab = {entry.key: entry.value for entry in model_proto.vocab}

    default_tokens = {
        "pad": model_proto.special_tokens.pad,
        "unk": model_proto.special_tokens.unk,
        "sos": model_proto.special_tokens.sos,
        "eos": model_proto.special_tokens.eos,
    }

    custom_tokens = (
        {entry.sp_token: entry.value for entry in model_proto.custom_tokens}
        if model_proto.custom_tokens
        else {}
    )

    normalizer_config = {
        "lower": model_proto.normalizer.lower,
        "unicode_format": model_proto.normalizer.unicode_format,
        "custom_rules": {
            rule.pattern: rule.replacement
            for rule in model_proto.normalizer.custom_rules
        },
    }

    model_type = model_proto.model_type
    vocab_size = model_proto.vocab_size

    if model_type == "unigram":
        unigram_config = {
            "max_sub_len": model_proto.unigram.max_sub_len,
            "em_iters": model_proto.unigram.em_iters,
            "n_per": model_proto.unigram.n_per,
            "epsilon": model_proto.unigram.epsilon,
        }
        unigram_prob = {pair.token: pair.prob for pair in model_proto.unigram_prob}
    elif model_type == "bpe":
        unigram_config = None
        unigram_prob = None
    else:
        raise ValueError("Unknown mode, Check model type again")

    return (
        vocab,
        default_tokens,
        custom_tokens,
        normalizer_config,
        model_type,
        vocab_size,
        unigram_config,
        unigram_prob,
    )


def load_vocab(path: str) -> Dict[str, int]:
    vocab = {}
    with open(path, "r") as fr:
        for line in fr:
            token, idx = line.strip().split("\t")
            vocab[token] = int(idx)

    return vocab
