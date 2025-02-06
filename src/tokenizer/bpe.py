from typing import List, Dict, Optional, Union
from collections import defaultdict
from tokenizer.meta_tokenizer import Tokenizer
from tokenizer.heap import MaxHeap
from tqdm import tqdm

class BPE(Tokenizer):
  def __init__(self, vocab_size: int= 50, special_token: Optional[Union[List[str], Dict[str, int]]] = None):
    self.vocab_size = vocab_size
    self.vocab = None
    self.pair_freq = defaultdict(int)
    self.char_sents = None
    self.heap = MaxHeap()

    if special_token:
      self.special_token = special_token
    else:
      self.special_token = list()


  def train(self, corpus: List[str]) -> Dict[str, int]:
      self._init_vocab(corpus)
      self._init_pair_freq()

      for pair, freq in self.pair_freq.items():
          self.heap.insert((freq, pair))

      try:
        pbar = tqdm(total=self.vocab_size, desc="BPE Training...", leave=False)
        while len(self.vocab) <= self.vocab_size - len(self.special_token) and len(self.heap) > 0:

            freq, pair = self.heap.extract_max()
            a, b = pair
            new_token = a + b
            self.vocab.add(new_token); pbar.update(1)
            self._update_pair_freq(pair, new_token)
            self.heap.update(self.pair_freq)

      except Exception as e:
        print(f"Error: {e}")

      finally:
        if isinstance(self.special_token, list):
          self.vocab = self.special_token + list(self.vocab)
          return {token: idx for idx, token in enumerate(self.vocab)}

        elif isinstance(self.special_token, dict):
          vocab_id = {token: idx for idx, token in enumerate(self.vocab)}

          overlap = 0
          for token, id in vocab_id.items():
              if id in self.special_token.values():
                  vocab_id[token] = len(vocab_id) + overlap
                  overlap += 1
          vocab_id.update(self.special_token)
          vocab_id = dict(sorted(vocab_id.items(), key=lambda x: x[1]))
          return vocab_id

        else:
          raise TypeError("self.special_token type must be List or Dictionary")

  def _init_vocab(self, corpus: List[str]):
    self.char_sents = [list(sent) for sent in corpus]
    self.vocab = set(sum(self.char_sents, []))

  def _init_pair_freq(self):
      """
      Create a frequency dictionary for all character pairs in the given sentences.

      Returns:
          Dict[Tuple[str, str], int]: A dictionary mapping character pairs to their frequency.
      """
      for sent in self.char_sents:
          for i in range(len(sent) - 1):
              pair = (sent[i], sent[i + 1])
              self.pair_freq[pair] += 1

  def _update_pair_freq(self, merged_pair, new_token):
    a, b = merged_pair
    for idx, sent in enumerate(self.char_sents):
        i = 0
        while i < len(sent) - 1:
            if sent[i] == a and sent[i + 1] == b:
                if i > 0:
                    prev_pair = (sent[i - 1], a)
                    self.pair_freq[prev_pair] -= 1
                    if self.pair_freq[prev_pair] == 0:
                        del self.pair_freq[prev_pair]
                    new_prev_pair = (sent[i - 1], new_token)
                    self.pair_freq[new_prev_pair] = self.pair_freq.get(new_prev_pair, 0) + 1

                if i < len(sent) - 2:
                    next_pair = (b, sent[i + 2])
                    self.pair_freq[next_pair] -= 1
                    if self.pair_freq[next_pair] == 0:
                        del self.pair_freq[next_pair]
                    new_next_pair = (new_token, sent[i + 2])
                    self.pair_freq[new_next_pair] = self.pair_freq.get(new_next_pair, 0) + 1

                self.pair_freq[(a, b)] -= 1
                if self.pair_freq[(a, b)] == 0:
                    del self.pair_freq[(a, b)]

                sent[i] = new_token
                del sent[i + 1]
            else:
                i += 1
            self.char_sents[idx] = sent

  def tokenize(self, normalized_sent: str) -> List[str]:
    tokens = []
    while len(normalized_sent) > 0:
      i = len(normalized_sent)
      while i > 0 and normalized_sent[:i] not in self.vocab:
        i -= 1
      if i == 0:
        pass  # unk token return
      tokens.append(normalized_sent[:i])
      normalized_sent = normalized_sent[i:]

    return tokens

  def detokenize(self, tokens: List[str]) -> str:
    return "".join(tokens).replace("▁", " ").strip()