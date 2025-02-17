# TODO: apply trie data structure
from typing import List, Dict
from collections import defaultdict
from tokenizer.meta_tokenizer import Tokenizer
from tqdm import tqdm
import math


def safe_log_prob(p: float, epsilon: float = 1e-10):
    return -math.log(max(p, epsilon))


class Unigram(Tokenizer):
    def __init__(
        self,
        vocab_size: int = 1000,
        unk_token: str = "[UNK]",
        special_token: Dict[str, int] = {"[UNK]": 0},
        vocab_seed: int = 4,
        max_sub_len: int = 8,
        em_iters: int = 3,
        n_per: float = 0.8,
        epsilon: float = 1e-10
    ):
        self.target_size = vocab_size
        self.vocab_seed = vocab_seed   # initial vocab seed size
        self.max_sub_len = max_sub_len
        self.em_iters = em_iters
        self.n_per = n_per  # remaining percentage
        self.vocab = None
        self.probs = dict()  # token probablity
        self.counts = None  # token frequency
        self.single_tokens = set()  # single character token (forbidding OOV)
        self.epsilon = epsilon  # forbidding 0 division error & unk_penalty

        self.unk_token = unk_token
        self.special_token = special_token 
    
    def train(self, corpus: List[str]) -> Dict[str, int]:
        """train vocab by unigram model

        Args:
            corpus (List[str]): Input Sentences

        Returns:
            Dict[str, int]: trained vocab
        """
        self._init_seed_vocab(corpus)

        pbar = tqdm(total=self.target_size, desc="Unigram Training...", ncols=100, position=0, leave=False)
        while len(self.vocab) >= self.target_size - len(self.special_token):
            self._run_em(corpus)

            losses = self._compute_losses()

            self._prune_vocab(losses)
            pbar.update(1)

        self._normalize_probs()

        # indexing vocab
        vocab_id = {token: idx for idx, token in enumerate(self.vocab)}

        overlap = 0
        for token, id in vocab_id.items():
            if id in self.special_token.values():
                vocab_id[token] = len(vocab_id) + overlap  # avoid index duplication
                overlap += 1
        vocab_id.update(self.special_token)
        vocab_id = dict(sorted(vocab_id.items(), key=lambda x: x[1]))
        return vocab_id

    def _init_seed_vocab(self, corpus: List[str]):
        """Initialize vocab and token probablity

        Args:
            corpus (List[str]): Input sentences
        """
        char_counts = defaultdict(int)
        subword_counts = defaultdict(int)

        for sent in corpus:
            chars = list(sent)
            for c in chars:
                char_counts[c] += 1

            for i in range(len(sent)):
                for j in range(i + 1, min(i + self.max_sub_len + 1, len(sent) + 1)):
                    subword = sent[i:j]
                    subword_counts[subword] += 1

        self.single_tokens = set(char_counts.keys())
        self.vocab = list(self.single_tokens)

        seed_size = self.target_size * self.vocab_seed  # adjust initial vocab size
        sorted_subs = sorted(subword_counts.items(), key=lambda x: -x[1])

        for sub, _ in sorted_subs:
            if len(self.vocab) >= seed_size:
                break
            if len(sub) > 1:
                self.vocab.append(sub)

        self.probs = {token: 1.0 / len(self.vocab) for token in self.vocab}

        self._normalize_probs()

    def _run_em(self, corpus: List[str]):
        """run EM algorithm

        Args:
            corpus (List[str]): Input sentences
        """
        for _ in range(self.em_iters):
            # E-step: caculate expectation value
            self.counts = defaultdict(int)
            total = 0

            for sent in corpus:
                tokens = self._viterbi_segment(sent)
                for token in tokens:
                    self.counts[token] += 1
                    total += 1

            # M-step: update probabilities
            if total == 0:
                continue

            total = sum(self.counts.values()) + self.epsilon
            self.probs = {
                token: (
                    self.counts.get(token, 0) + self.epsilon / total
                )  # avoid zero for log calculation
                for token in self.vocab
            }
            self._normalize_probs()

    def _compute_losses(self) -> Dict[str, float]:
        """Compute losses when token is removed

        Returns:
            Dict[str, float]: computed losses
        """
        losses = {}
        for token in self.vocab:
            if token in self.counts:
                losses[token] = self.counts[token] * safe_log_prob(
                    self.probs[token], self.epsilon
                )  # frequency * log probability
            else:
                losses[token] = float("-inf")

        return losses

    def _prune_vocab(self, losses: Dict[str, float]):
        """prune vocab by n_per

        Args:
            losses (Dict[str, float]): token losses
        """
        multi_tokens = [t for t in self.vocab if len(t) > 1]
        single_tokens = [t for t in self.vocab if len(t) == 1]

        sorted_multi = sorted(multi_tokens, key=lambda x: -losses[x])

        keep_size = int(len(sorted_multi) * self.n_per)
        kept_multi = sorted_multi[:keep_size]

        new_vocab = single_tokens + kept_multi
        self.vocab = new_vocab

        self.probs = {token: self.probs[token] for token in new_vocab}
        self._normalize_probs()

    def _normalize_probs(self):
        """normalize probality"""
        total = sum(self.probs.values())
        if total == 0:
            return
        self.probs = {k: v / total for k, v in self.probs.items()}

    def _viterbi_segment(self, sent: str) -> List[str]:
        """find best segmentation of sentence by viterbi algorithm

        Args:
            sent (str): Input sentence

        Returns:
            List[str]: Segmented sentence
        """
        n = len(sent)
        dp = [float("-inf")] * (n + 1)
        dp[0] = 0
        backpointer = [[] for _ in range(n + 1)]

        for i in range(n + 1):
            for j in range(max(0, i - self.max_sub_len), i):
                token = sent[j:i]
                if token in self.probs:
                    prob = safe_log_prob(self.probs.get(token), self.epsilon)

                    if dp[j] + prob > dp[i]:
                        dp[i] = dp[j] + prob
                        backpointer[i] = backpointer[j] + [
                            token
                        ]  # if probabiltiy is bigger, save segmentation

                else:
                    if not backpointer[i]:
                        backpointer[i] = backpointer[j] + [self.unk_token]

        return backpointer[n] if dp[n] != float("-inf") else [sent]

    def tokenize(self, normalized_text: str) -> List[str]:
        """tokenize normalized sentence with vocab

        Args:
            normalized_text (str): sentence to tokenize

        Returns:
            List[str]: tokenized sentence
        """
        return self._viterbi_segment(normalized_text)

    def detokenize(self, tokens: List[str]) -> str:
        """detokenize tokens to sentence

        Args:
            tokens (List[str]): Token to detokenize

        Returns:
            str: Detokenized sentence
        """
        return "".join(tokens).replace("▁", " ").strip()
