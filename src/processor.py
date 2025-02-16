from typing import List
from normalizer import Normalizer
from tokenizer.bpe import BPE
from tokenizer.unigram import Unigram
from utils import load_model


class Processor:
    def __init__(self, model_path: str):
        (
            vocab,
            default_tokens,
            custom_tokens,
            normalizer_config,
            mode,
            vocab_size,
            unigram_config,
            unigram_prob,
        ) = load_model(model_path)
        self.word2idx = vocab
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.normalizer = Normalizer(**normalizer_config)

        self.default_tokens = default_tokens
        self.custom_tokens = custom_tokens
        self.special_tokens = {
            entry.token: entry.value for entry in default_tokens.values()
        }.update(custom_tokens)

        if mode == "bpe":
            self.tokenizer = BPE(
                vocab_size,
                unk_token=default_tokens["unk"].token,
                special_token=self.special_tokens,
            )
        elif mode == "unigram":
            self.tokenizer = (
                Unigram(
                    vocab_size,
                    unk_token=self.unk_token,
                    special_token=self.special_token,
                    **unigram_config,
                )
                if unigram_config
                else Unigram(vocab_size, unk_token=self.unk_token, special_token=self.special_token)
            )
            self.tokenizer.probs = unigram_prob
        else:
            raise ValueError("mode must be BPE or Unigram")
        self.tokenizer.vocab = self.word2idx

    def get_piece_size(self) -> int:
        """get vocab size

        Returns:
            int: vocab size
        """
        return len(self.word2idx)

    def id_to_piece(self, id: int) -> str:
        """convert id into subword

        Args:
            id (int): id to convert

        Returns:
            str: converted subwords
        """
        return self.idx2word[id]

    def piece_to_id(self, piece: str) -> int:
        """convert subword into id

        Args:
            piece (str): subword pieces to convert

        Returns:
            int: converted ids
        """
        return self.word2idx[piece]

    def decode_pieces(self, subwords: List[str]) -> str:
        """convert list of subwords into sentence

        Args:
            subwords (List[str]): subword list to convert

        Returns:
            str: converted sentence
        """
        return self.tokenizer.detokenize(subwords)

    def decode_ids(self, ids: List[int]) -> str:
        """convert list of ids into sentence

        Args:
            ids (List[int]): id list to convert

        Returns:
            str: converted sentence
        """
        subwords = [self.id_to_piece(id) for id in ids]

        return self.decode_pieces(subwords)

    def encode(self, sentence: str, option: type = str) -> List[str] | List[int]:
        """convert sentence to subwords or ids

        Args:
            sentence (str): sentence to convert
            option (type, optional): id or suword setting. Defaults to str.

        Raises:
            ValueError: option type error

        Returns:
            List[str] | List[int]: converted subwords or ids
        """
        sentence = self.normalizer.normalize(sentence)
        subwords = self.tokenizer.tokenize(sentence)

        if option == str:
            return subwords
        elif option == int:
            return [self.piece_to_id(subword) for subword in subwords]
        else:
            raise ValueError("option must be str or int")

    def encode_with_sp_tokens(
        self, sentence: str, max_length: int, option: type = str
    ) -> List[str] | List[int]:
        """convert sentence to suwords or ids with special tokens

        Args:
            sentence (str): sentence to convert
            max_length (int): padding length
            option (type, optional): id or suword setting. Defaults to str.

        Raises:
            ValueError: option type error

        Returns:
            List[str] | List[int]: converted subwords or ids
        """
        subwords = self.encode(sentence, int)
        encoded = [self.special_tokens["sos"]] + \
            subwords + [self.special_tokens["eos"]]

        if len(encoded) < max_length:
            encoded += [self.special_tokens["pad"]] * \
                (max_length - len(encoded))
        else:
            encoded = encoded[:max_length]

        if option == str:
            return [self.id_to_piece(id) for id in encoded]
        elif option == int:
            return encoded
        else:
            raise ValueError("option must be str or int")

    def decode_ids_with_sp_tokens(self, ids: List[int]) -> str:
        """convert ids with special tokens to sentence

        Args:
            ids (List[int]): ids to convert

        Returns:
            str: converted sentence
        """
        filtered_ids = [
            id for id in ids if id not in self.special_tokens.values()]
        return self.decode_ids(filtered_ids)

    def decode_pieces_with_sp_tokens(self, pieces: List[str]) -> str:
        """convert subwords with special tokens to sentence

        Args:
            pieces (List[str]): subwords to convert

        Returns:
            str: converted sentence
        """
        filitered_pieces = [
            piece for piece in pieces if piece not in self.special_tokens.keys()
        ]
        return self.decode_pieces(filitered_pieces)
