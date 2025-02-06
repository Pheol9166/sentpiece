from typing import Dict
from model_pb2 import ModelProto
import json

def load_json(path: str):
  with open(path, 'r') as json_file:
    return json.load(json_file)


def load_model(path: str):
  model_proto = ModelProto()
  with open(path, "rb") as fr:
    model_proto.ParseFromString(fr.read())
  
  vocab = {entry.key: entry.value for entry in model_proto.vocab}
  
  special_tokens = []
  for token_entry in model_proto.special_tokens:
    if token_entry.HasFile("token_str"):
      special_tokens.append(token_entry.token_str)
    elif token_entry.HasFile("token_pair"):
      special_tokens.append((token_entry.token_pair.key, token_entry.token_pair.value))
    else:
      raise TypeError("Unknown token type")
  
  normalizer_config = {
    "lower": model_proto.normalizer.lower,
    "unicode_format": model_proto.normalizer.unicode_format,
    "custom_rules": {rule.pattern: rule.replacement for rule in model_proto.normalizer.custom_rules}
  }

  model_type = model_proto.model_type
  vocab_size = model_proto.vocab_size

  if model_type == "unigram":
    unigram_config = {
      "max_sub_len": model_proto.unigram.max_sub_len,
      "em_iters": model_proto.unigram.em_iters,
      "n_per": model_proto.unigram.n_per,
      "epsilon": model_proto.unigram.epsilon,
      "alpha": model_proto.unigram.alpha,
      "unk_penalty": model_proto.unigram.unk_penalty
    }
  elif model_type == "bpe":
    unigram_config = None
  else:
    raise ValueError("Unknown mode, Check model type again")

  return vocab, special_tokens, normalizer_config, model_type, vocab_size, unigram_config 

def load_vocab(path: str) -> Dict[str, int]:
  vocab = {}
  with open(path, "r") as fr:
    for line in fr:
      token, idx = line.strip().split("\t")
      vocab[token] = int(idx)

  return vocab