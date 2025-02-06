from typing import List
from normalizer import Normalizer
from tokenizer.bpe import BPE
from tokenizer.unigram import Unigram
from utils import load_model

class Processor:
  def __init__(self, model_path: str):
    vocab, special_tokens, normalizer_config, mode, vocab_size, unigram_config = load_model(model_path)
    self.word2idx = vocab
    self.idx2word = {idx: word for word, idx in self.word2idx.items()}
    self.normalizer = Normalizer(**normalizer_config)
    
    if isinstance(special_tokens[0], tuple):
      special_tokens = {token: idx for token, idx in special_tokens}

    if mode == "bpe":
      self.tokenizer = BPE(vocab_size, special_tokens)
    elif mode == "unigram":
      self.tokenizer = Unigram(vocab_size, special_token=special_tokens, **unigram_config)
    else:
      raise ValueError("mode must be BPE or Unigram")
    self.tokenizer.vocab = self.word2idx

  def get_piece_size(self) -> int:
    return len(self.word2idx)

  def id_to_piece(self, id: int) -> str:
    """
    convert id into subword
    """
    return self.idx2word[id]
    # return self.special_token   아직까지는 확정 x

  def piece_to_id(self, piece: str) -> int:
    """
    convert subword into id
    """
    return self.word2idx[piece]

  def decode_pieces(self, subwords: List[str]) -> str:
    """
    convert list of subwords into sentence
    """
    return self.tokenizer.detokenize(subwords)

  def decode_ids(self, ids: List[int]) -> str:
    """
    convert list of ids into sentence
    """
    subwords = [self.id_to_piece(id) for id in ids]

    return self.decode_pieces(subwords)

  def encode(self, sentence: str, option: type=str) -> List[str]:
    sentence = self.normalizer.normalize(sentence)
    subwords = self.tokenizer.tokenize(sentence)

    if option == str:
      return subwords
    elif option == int:
      return [self.piece_to_id(subword) for subword in subwords]
    else:
      raise ValueError("option must be str or int")
