from typing import List, Dict, Tuple
from collections import defaultdict
from tokenizer.meta_tokenizer import Tokenizer
from tokenizer.heap import MaxHeap
from tqdm import tqdm
from multiprocessing import Pool


class BPE(Tokenizer):
    def __init__(
        self,
        vocab_size: int = 50,
        unk_token: str = "[UNK]",
        special_token: Dict[str, int] = {"[UNK]": 0},
    ):
        self.vocab_size = vocab_size
        self.vocab = None
        self.pair_freq = defaultdict(int)
        self.char_sents = None
        self.heap = MaxHeap()
        self.unk_token = unk_token
        self.special_token = special_token

    def train(self, corpus: List[str]) -> Dict[str, int]:
        """train vocab by bpe model

        Args:
            corpus (List[str]): Input sentences

        Returns:
            Dict[str, int]: trained vocab
        """
        self._init_vocab(corpus)
        self._init_pair_freq()

        for pair, freq in self.pair_freq.items():
            self.heap.insert((freq, pair))

        pbar = tqdm(
            total=self.vocab_size,
            desc="BPE Training...",
            ncols=100,
            position=0,
            leave=True,
        )

        while (
            len(self.vocab) < self.vocab_size - len(self.special_token)
            and len(self.heap) > 0
        ):
            result = self.heap.extract_max()
            if result is None:
                break

            freq, pair = result
            current_freq = self.pair_freq.get(pair, 0)

            if current_freq == freq and current_freq > 0:
                a, b = pair
                new_token = a + b
                self.vocab.add(new_token)

                # update pair frequency
                self._update_pair_freq(pair, new_token)

                self._init_pair_freq()

                self.heap.update(self.pair_freq)

                pbar.update(1)

        vocab_id = {token: idx for idx, token in enumerate(self.vocab)}

        # special token handling
        overlap = 0
        for token, id in vocab_id.items():
            if id in self.special_token.values():
                vocab_id[token] = len(vocab_id) + overlap
                overlap += 1
        vocab_id.update(self.special_token)

        vocab_id = dict(sorted(vocab_id.items(), key=lambda x: x[1]))

        return vocab_id

    def _init_vocab(self, corpus: List[str]):
        """Initialize vocab in given corpus

        Args:
            corpus (List[str]): Input sentences
        """
        self.char_sents = list()
        vocab = set()
        for sent in corpus:
            chars = list(sent)
            self.char_sents.append(chars)
            vocab.update(chars)
        self.vocab = vocab

    def _compute_pair_freq(sent):
        freq = defaultdict(int)
        for i in range(len(sent) - 1):
            pair = (sent[i], sent[i + 1])
            freq[pair] += 1
        return freq

    def _init_pair_freq(self):
        """Create a frequency dictionary for all character pairs in the given sentences."""

        with Pool() as pool:
            results = pool.map(BPE._compute_pair_freq, self.char_sents)

        new_pair_freq = defaultdict(int)

        for pair_freq in results:
            for pair, freq in pair_freq.items():
                new_pair_freq[pair] += freq

        self.pair_freq = new_pair_freq

    def _process_pair_freq(args):
        sent, a, b, new_token = args
        new_sent = []
        i = 0
        while i < len(sent):
            if i < len(sent) - 1 and sent[i] == a and sent[i + 1] == b:
                new_sent.append(new_token)
                i += 2
            else:
                new_sent.append(sent[i])
                i += 1
        return new_sent

    def _update_pair_freq(self, merged_pair: Tuple[str, str], new_token: str):
        """Update pair frequency for all character pairs when given pair is merged

        Args:
            merged_pair (Tuple[str, str]): Pair to merge
            new_token (str): Merged result
        """
        a, b = merged_pair
        with Pool() as p:
            args = [(sent, a, b, new_token) for sent in self.char_sents]
            self.char_sents = p.map(BPE._process_pair_freq, args)

    def tokenize(self, normalized_sent: str) -> List[str]:
        """tokenize normalized sentence with trained vocab

        Args:
            normalized_sent (str): Sentence to tokenize
        Returns:
            List[str]: Tokenized result
        """
        tokens = []
        while len(normalized_sent) > 0:
            i = len(normalized_sent)
            while i > 0 and normalized_sent[:i] not in self.vocab:
                i -= 1
            if i == 0:
                tokens.append(self.unk_token)
            tokens.append(normalized_sent[:i])
            normalized_sent = normalized_sent[i:]

        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        """detokenize tokens to sentence

        Args:
            tokens (List[str]): Token to detokenize

        Returns:
            str: Detokenized sentence
        """
        return "".join(tokens).replace("▁", " ").strip()
