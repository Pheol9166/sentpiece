from typing import List, Dict, Union, Any, Optional
from model_pb2 import ModelProto
from tokenizer.bpe import BPE
from tokenizer.unigram import Unigram
from normalizer import Normalizer


# special_token 수정, trainer에서 special token을 주니까, 따로 클래스를 만들던가 아니면 dict를 활용해서 tokenizer 모델들에게는 special token 변수 하나만 받게 하면 될듯
# self.special_token을 할 필요가 있을까?
class Trainer:
    def __init__(
        self,
        vocab_size: int = 100,
        model_prefix: str = "spm",
        mode: str = "unigram",
        pad_token: str = "[PAD]",
        pad_id: int = 0,
        unk_token: str = "[UNK]",
        unk_id: int = 1,
        sos_token: str = "[SOS]",
        sos_id: int = 2,
        eos_token: str = "[EOS]",
        eos_id: int = 3,
        custom_special_tokens: Optional[Dict[str, int]] = None,
        normalizer_config: Optional[Dict[str, Any]] = None,
        unigram_config: Optional[Dict[str, Any]] = None
    ):
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        self.special_token = {
            pad_token: pad_id,
            unk_token: unk_id,
            sos_token: sos_id,
            eos_token: eos_id,
        }
        self.custom_tokens = custom_special_tokens
        if self.custom_tokens:
            self.special_token.update(self.custom_special_tokens)

        self.normalizer = (
            Normalizer(**normalizer_config) if normalizer_config else Normalizer()
        )
        self.mode = mode

        if mode.lower() == "bpe":
            self.tokenizer = BPE(vocab_size, unk_token=self.unk_token, special_token=self.special_token)
        elif mode.lower() == "unigram":
            self.tokenizer = (
                Unigram(vocab_size, unk_token=self.unk_token, special_token=self.special_token, **unigram_config)
                if unigram_config
                else Unigram(vocab_size, special_token=self.special_token)
            )
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
        
        default_tokens = model_proto.special_tokens
        default_tokens.pad.token = self.pad_token
        default_tokens.unk.token = self.unk_token
        default_tokens.sos.token = self.sos_token
        default_tokens.eos.token = self.eos_token
        
        default_tokens.pad.value = self.special_token[self.pad_token]
        default_tokens.unk.value = self.special_token[self.unk_token]
        default_tokens.sos.value = self.special_token[self.sos_token]
        default_tokens.eos.value = self.special_token[self.eos_token]
        
        if self.custom_tokens:
            for token, idx in self.custom_tokens.items():
                entry = model_proto.custom_tokens.add()
                entry.sp_token = token
                entry.value = idx

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

            for token, prob in self.tokenizer.probs.items():
                pair = model_proto.unigram_prob.add()
                pair.token = token
                pair.prob = prob

        with open(f"{self.model_prefix}.pb", "wb") as fw:
            fw.write(model_proto.SerializeToString())

    def train(self, corpus: List[str]) -> Dict[str, int]:
        corpus = [self.normalizer.normalize(sent) for sent in corpus]
        return self.tokenizer.train(corpus)
