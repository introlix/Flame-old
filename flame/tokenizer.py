import os
from logging import getLogger
from typing import (
    Dict,
    List,
    Sequence,
    cast,
)

logger = getLogger(__name__)

import tiktoken

class Tokenizer:
    """
    Tokenizing text using the Tiktoken tokenizer.
    """
    special_tokens: Dict[str, int]

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    def __init__(self):
        """
        Initializes the Tokenizer with a Tiktoken model.
        """
        cl100k_base = tiktoken.get_encoding("cl100k_base")
        num_base_tokens = len(cl100k_base._mergeable_ranks)

        special_tokens = [
            "<|start_of_seq|>",
            "<|end_of_seq|>",
        ]

        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }

        self.model = tiktoken.Encoding(
            name="cl100k_im",
            pat_str=self.pat_str,
            mergeable_ranks=cl100k_base._mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        logger.info(f"Reloaded tiktoken model")

        self.n_words: int = self.model.n_vocab
        
        self.sos_id: int = self.special_tokens["<|start_of_seq|>"]
        self.eos_id: int = self.special_tokens["<|end_of_seq|>"]
        self.pad_id: int = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_seq|>"],
        }

        logger.info(f"No. of words: {self.n_words}, SOS_ID: {self.sos_id}, and EOS_ID: {self.eos_id}")


    def encode(self, seq:str, sos:bool, eos:bool) -> List:
        """
        Encoding string to token IDs

        Args:
            seq (str): The input string to be encoded.
            sos (bool): Whether to add the sos token.
            eos (bool): Whether to add the eos token.

        Returns:
            list[int]: A list of token IDs.
        """
            
        token_ids: List[int] = []

        token_ids.extend(self.model.encode(seq))

        if sos:
            token_ids.insert(0, self.sos_id)
        if eos:
            token_ids.append(self.eos_id)

        return token_ids
    
    def decode(self, token_ids:Sequence[int]) -> str:
        """
        Decoding token IDs to string

        Args:
            token_ids (List[int]): The list of token ids to be decoded

        Returns:
            seq (str): The decoded string
        """

        return self.model.decode(cast(List[int], token_ids))
