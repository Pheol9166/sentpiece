from typing import List, Dict
from abc import ABCMeta, abstractmethod

class Tokenizer(metaclass=ABCMeta):
  @abstractmethod
  def train(self, corpus: List[str]) -> Dict[str, int]:
    pass

  @abstractmethod
  def tokenize(self, sent: str) -> List[str]:
    pass

  @abstractmethod
  def detokenize(self, tokens: List[str]) -> str:
    pass