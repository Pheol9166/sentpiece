from typing import List, Dict, Union, Any, Optional
from model_pb2 import ModelProto
from tokenizer.bpe import BPE
from tokenizer.unigram import Unigram
from normalizer import Normalizer

class Trainer:
  def __init__(self, vocab_size: int= 100, model_prefix: str= "spm", mode: str="unigram", special_token: Union[List, Dict]= ["[UNK]"], normalizer_config: Optional[Dict[str, Any]]=None, unigram_config: Optional[Dict[str, Any]]= None, unigram_prob: Optional[Dict[str, float]]= None):
    self.vocab_size = vocab_size
    self.model_prefix = model_prefix
    self.special_token = special_token
    self.normalizer = Normalizer(**normalizer_config) if normalizer_config else Normalizer()
    self.mode = mode

    if mode.lower() == "bpe":
      self.tokenizer = BPE(vocab_size, special_token=special_token)
    elif mode.lower() == "unigram":
      self.tokenizer = Unigram(vocab_size, special_token=self.special_token,  **unigram_config) if unigram_config else Unigram(vocab_size, special_token=self.special_token)
      self.tokenizer.probs = unigram_prob
    else:
      raise ValueError("mode must be bpe or unigram")

  def save_vocab(self, vocab: Dict[str, int]):
    with open(f"./{self.model_prefix}_vocab.txt", "w") as fw:
      for token, idx in vocab.items():
        fw.write(f"{token}\t{idx}\n")

  def save(self, vocab: Dict[str, int]):
    self.save_vocab(vocab)  # save vocab

    model_proto = ModelProto()  # ModelProto in model proto file

    for key, value in vocab.items():
      entry = model_proto.vocab.add()
      entry.key = key
      entry.value = value

    if isinstance(self.special_token, list):
      for token in self.special_token:
        entry = model_proto.special_tokens.add()
        entry.token_str = token
    elif isinstance(self.special_token, dict):
      for token, idx in self.special_token.items():
        entry = model_proto.special_tokens.add()
        entry.token_pair.key = token
        entry.token_pair.value = idx
    else:
      raise ValueError("special_token type must be List or Dictionary")

    normalizer_config = model_proto.normalizer
    normalizer_config.lower = self.normalizer.lower
    normalizer_config.unicode_format = self.normalizer.unicode_format

    for ptn, repl in self.normalizer.custom_rules.items():
      custom_rule = normalizer_config.custom_rules.add()
      custom_rule.pattern = ptn
      custom_rule.replacement = repl

    model_proto.model_type = self.mode

    model_proto.vocab_size = self.vocab_size

    if self.mode.lower() == "unigram":
      unigram_config = model_proto.unigram
      unigram_config.max_sub_len = self.tokenizer.max_sub_len
      unigram_config.em_iters = self.tokenizer.em_iters
      unigram_config.n_per = self.tokenizer.n_per
      unigram_config.epsilon = self.tokenizer.epsilon
      unigram_config.alpha = self.tokenizer.alpha
      unigram_config.unk_penalty = self.tokenizer.unk_penalty

      for token, prob in self.tokenizer.probs.items():
        pair = model_proto.unigram_prob.add()
        pair.token = token
        pair.prob = prob

    with open(f"{self.model_prefix}.pb", "wb") as fw:
      fw.write(model_proto.SerializeToString())

  def train(self, corpus: List[str]) -> Dict[str, int]:
    corpus = [self.normalizer.normalize(sent) for sent in corpus]
    return self.tokenizer.train(corpus)