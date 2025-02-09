from typing import Optional, Dict
import unicodedata
import re


class Normalizer:
    def __init__(
        self,
        lower: bool = True,
        unicode_format: str = "NFKC",
        custom_rules: Optional[Dict[str, str]] = None,
    ):
        """
        Text normalize class

        Args:
          lower(bool): wheter to use lower change, default is True.
          unicode_format(str): Unicode normalization format, default is "NFKC".
          custom_rules(Optional[Dict[str]]): custom rules for normalization, default is None. format is "pattern replacement".
        """
        self.lower = lower
        self.unicode_format = unicode_format
        self.custom_rules = custom_rules or {}

    def normalize(self, text: str) -> str:
        """
        normalize input text

        Args:
          text(str): input text

        Returns:
          str: normalized text
        """

        if self.unicode_format:
            text = unicodedata.normalize(self.unicode_format, text)

        if self.lower:
            text = text.lower()

        for pattern, replacement in self.custom_rules.items():
            text = re.sub(pattern, replacement, text)

        text = re.sub(r"\s+", " ", text).strip()

        text = re.sub(" ", "▁", text)

        return f"▁{text}"
