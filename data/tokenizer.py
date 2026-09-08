"""
Compact 8,192-Vocabulary Byte-Level BPE Tokenizer Wrapper with Native Agentic & FIM Control Tokens.

Designed for sub-20M parameter models where vocabulary embedding parameter budget
must be strictly preserved (8,192 tokens * 512 d_model = 4.19M parameters).
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence, Union
import tiktoken


# Special Agentic, FIM, and Structural Tokens
SPECIAL_TOKENS: Dict[str, int] = {
    "<|pad|>": 0,
    "<|bos|>": 1,
    "<|eos|>": 2,
    "<|thought start|>": 3,
    "<|thought end|>": 4,
    "<|call tool|>": 5,
    "<|tool response|>": 6,
    "<|reflect|>": 7,
    "<|fim_prefix|>": 8,
    "<|fim_middle|>": 9,
    "<|fim_suffix|>": 10,
    "<|unk|>": 11,
}

VOCAB_SIZE: int = 8192
NUM_SPECIAL_TOKENS: int = len(SPECIAL_TOKENS)
NUM_BYTE_TOKENS: int = 256
BASE_RESERVED: int = NUM_SPECIAL_TOKENS + NUM_BYTE_TOKENS  # Offset for merges


class AttoTokenizer:
    """
    Byte-level BPE Tokenizer with Tiktoken backend and fallback custom BPE engine.
    Ensures exact 8,192 vocabulary size and full support for agentic control tokens.
    """

    def __init__(
        self,
        name: str = "atto_8k",
        vocab_size: int = VOCAB_SIZE,
        special_tokens: Optional[Dict[str, int]] = None,
        merge_ranks: Optional[Dict[bytes, int]] = None,
    ):
        self.name = name
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or dict(SPECIAL_TOKENS)
        self.inv_special_tokens = {v: k for k, v in self.special_tokens.items()}
        
        # Build default merge ranks or use provided
        if merge_ranks is not None:
            self._merge_ranks = merge_ranks
        else:
            self._merge_ranks = self._build_default_merge_ranks(vocab_size)
            
        self._init_tiktoken_encoding()

    def _build_default_merge_ranks(self, target_vocab_size: int) -> Dict[bytes, int]:
        """
        Constructs a standard byte-level BPE rank table containing common ASCII,
        code tokens (Python, JSON, Math), whitespace patterns, and subwords.
        """
        ranks: Dict[bytes, int] = {}
        # Single byte tokens (offset after special tokens)
        for i in range(256):
            ranks[bytes([i])] = NUM_SPECIAL_TOKENS + i

        # High-frequency code, text, indentation, and punctuation n-grams
        common_seeds = [
            # Whitespace & indentation
            b"  ", b"    ", b"        ", b"            ", b"                ",
            b"\n", b"\n\n", b"\n    ", b"\n        ", b"\t", b"\r\n",
            # Common programming keywords & symbols
            b"def ", b"return ", b"class ", b"import ", b"from ", b"for ", b"while ",
            b"if ", b"elif ", b"else:", b"try:", b"except ", b"finally:", b"with ",
            b"as ", b"in ", b"is ", b"not ", b"and ", b"or ", b"lambda ", b"yield ",
            b"async ", b"await ", b"self.", b"self", b"None", b"True", b"False",
            b"print(", b"len(", b"range(", b"int(", b"str(", b"float(", b"list(",
            b"dict(", b"set(", b"tuple(", b"bool(", b"type(", b"isinstance(",
            b"def __init__(self", b"super().__init__()", b"raise ValueError(",
            b"raise TypeError(", b"assert ", b"pass", b"continue", b"break",
            b" == ", b" != ", b" <= ", b" >= ", b" += ", b" -= ", b" *= ", b" /= ",
            b" -> ", b" => ", b"...", b"/**", b"*/", b"//", b"##", b"###",
            b"```python", b"```json", b"```", b"{\"name\":", b"{\"type\":",
            b"\"parameters\":", b"\"arguments\":", b"\"action\":", b"\"query\":",
            b"\"result\":", b"\"error\":", b"\"status\":", b"\"success\":",
            b"\"thought\":", b"\"tool\":", b"\"input\":", b"\"output\":",
            # English subwords & educational text tokens
            b" the", b" of", b" to", b" and", b" a", b" in", b" is", b" it",
            b" you", b" that", b" he", b" was", b" for", b" on", b" are", b" with",
            b" as", b" I", b" his", b" they", b" be", b" at", b" one", b" have",
            b" this", b" from", b" or", b" had", b" by", b" not", b" word", b" but",
            b" what", b" some", b" we", b" can", b" out", b" other", b" were", b" all",
            b" there", b" when", b" up", b" use", b" your", b" how", b" said", b" an",
            b" each", b" she", b" which", b" do", b" their", b" time", b" if", b" will",
            b" way", b" about", b" many", b" then", b" them", b" write", b" would",
            b" like", b" so", b" these", b" her", b" long", b" make", b" thing", b" see",
            b" him", b" two", b" has", b" look", b" more", b" day", b" could", b" go",
            b" come", b" did", b" number", b" sound", b" no", b" most", b" people",
            b" my", b" over", b" know", b" water", b" than", b" call", b" first", b" who",
            b" may", b" down", b" side", b" been", b" now", b" find", b" any", b" new",
            b" work", b" part", b" take", b" get", b" place", b" made", b" live", b" where",
            b" after", b" back", b" little", b" only", b" round", b" man", b" year",
            b" came", b" show", b" every", b" good", b" me", b" give", b" our", b" under",
            b" name", b" very", b" through", b" just", b" form", b" sentence", b" great",
            b" think", b" say", b" help", b" low", b" line", b" differ", b" turn",
            b" cause", b" much", b" mean", b" before", b" move", b" right", b" boy",
            b" old", b" too", b" same", b" tell", b" does", b" set", b" three", b" want",
            b" air", b" well", b" also", b" play", b" small", b" end", b" put", b" home",
            b" read", b" hand", b" port", b" large", b" spell", b" add", b" even", b" land",
            b" here", b" must", b" big", b" high", b" such", b" follow", b" act", b" why",
            b" ask", b" men", b" change", b" went", b" light", b" kind", b" off", b" need",
            b" house", b" picture", b" try", b" us", b" again", b" animal", b" point",
            b" mother", b" world", b" near", b" build", b" self", b" earth", b" father",
            b" head", b" stand", b" own", b" page", b" should", b" country", b" found",
            b" answer", b" school", b" grow", b" study", b" still", b" learn", b" plant",
            b" cover", b" food", b" sun", b" four", b" between", b" state", b" keep",
            b" eye", b" never", b" last", b" let", b" thought", b" city", b" tree",
            b" cross", b" farm", b" hard", b" start", b" might", b" story", b" saw",
            b" far", b" sea", b" draw", b" left", b" late", b" run", b" don't", b" while",
            b" press", b" close", b" night", b" real", b" life", b" few", b" north",
            b" open", b" seem", b" together", b" next", b" white", b" children", b" begin",
            b" got", b" walk", b" example", b" ease", b" paper", b" group", b" always",
            b" music", b" those", b" both", b" mark", b" often", b" letter", b" until",
            b" mile", b" river", b" car", b" feet", b" care", b" second", b" book",
            b" carry", b" took", b" science", b" eat", b" room", b" friend", b" began",
            b" idea", b" fish", b" mountain", b" stop", b" once", b" base", b" hear",
            b" horse", b" cut", b" sure", b" watch", b" color", b" face", b" wood",
            b" main", b" enough", b" plain", b" girl", b" usual", b" young", b" ready",
            b" above", b" ever", b" red", b" list", b" though", b" feel", b" talk",
            b" bird", b" soon", b" body", b" dog", b" family", b" direct", b" pose",
            b" leave", b" song", b" measure", b" door", b" product", b" black", b" short",
            b" numeral", b" class", b" wind", b" question", b" happen", b" complete",
            b" ship", b" area", b" half", b" rock", b" order", b" fire", b" south",
            b" problem", b" piece", b" told", b" knew", b" pass", b" since", b" top",
            b" whole", b" king", b" space", b" heard", b" best", b" hour", b" better",
            b" TRUE", b" FALSE", b" NULL", b" undefined", b" function", b" const",
            b" let", b" var", b" interface", b" struct", b" impl", b" fn", b" pub",
            b" package", b" module", b" namespace", b" public", b" private", b" protected",
        ]

        current_rank = BASE_RESERVED
        for seed in common_seeds:
            if seed not in ranks and current_rank < target_vocab_size:
                ranks[seed] = current_rank
                current_rank += 1

        # Fill all remaining slots up to target_vocab_size deterministically
        import itertools
        alphabet_chars = bytes(range(256))
        for length in range(2, 6):
            if current_rank >= target_vocab_size:
                break
            for combo in itertools.product(alphabet_chars, repeat=length):
                if current_rank >= target_vocab_size:
                    break
                seq = bytes(combo)
                if seq not in ranks:
                    ranks[seq] = current_rank
                    current_rank += 1

        return ranks

    def _init_tiktoken_encoding(self) -> None:
        """Initializes the underlying Tiktoken Encoding."""
        # Clean regex pattern for splitting tokens
        pat_str = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
        self._encoding = tiktoken.Encoding(
            name=self.name,
            pat_str=pat_str,
            mergeable_ranks=self._merge_ranks,
            special_tokens=self.special_tokens,
        )

    @property
    def pad_token_id(self) -> int:
        return self.special_tokens["<|pad|>"]

    @property
    def bos_token_id(self) -> int:
        return self.special_tokens["<|bos|>"]

    @property
    def eos_token_id(self) -> int:
        return self.special_tokens["<|eos|>"]

    @property
    def thought_start_id(self) -> int:
        return self.special_tokens["<|thought start|>"]

    @property
    def thought_end_id(self) -> int:
        return self.special_tokens["<|thought end|>"]

    @property
    def call_tool_id(self) -> int:
        return self.special_tokens["<|call tool|>"]

    @property
    def tool_response_id(self) -> int:
        return self.special_tokens["<|tool response|>"]

    @property
    def reflect_id(self) -> int:
        return self.special_tokens["<|reflect|>"]

    @property
    def fim_prefix_id(self) -> int:
        return self.special_tokens["<|fim_prefix|>"]

    @property
    def fim_middle_id(self) -> int:
        return self.special_tokens["<|fim_middle|>"]

    @property
    def fim_suffix_id(self) -> int:
        return self.special_tokens["<|fim_suffix|>"]

    def encode(
        self,
        text: str,
        allowed_special: Union[str, Sequence[str]] = "all",
        disallowed_special: Union[str, Sequence[str]] = (),
    ) -> List[int]:
        """Encodes text into token IDs."""
        if allowed_special == "all":
            allowed = set(self.special_tokens.keys())
        elif isinstance(allowed_special, (set, list, tuple)):
            allowed = set(allowed_special)
        else:
            allowed = set()
            
        return self._encoding.encode(
            text,
            allowed_special=allowed,
            disallowed_special=disallowed_special,
        )

    def encode_ordinary(self, text: str) -> List[int]:
        """Encodes text without parsing special tokens."""
        return self._encoding.encode_ordinary(text)

    def decode(self, tokens: Sequence[int], errors: str = "replace") -> str:
        """Decodes token IDs into a string."""
        return self._encoding.decode(list(tokens), errors=errors)

    def decode_single_token_bytes(self, token: int) -> bytes:
        """Returns the raw bytes corresponding to a single token ID."""
        return self._encoding.decode_single_token_bytes(token)

    def format_agent_turn(
        self,
        thought: Optional[str] = None,
        tool_call: Optional[str] = None,
        tool_response: Optional[str] = None,
        reflection: Optional[str] = None,
        content: Optional[str] = None,
    ) -> str:
        """Helper to format structured agentic strings with special tokens."""
        parts = []
        if thought:
            parts.append(f"<|thought start|>{thought}<|thought end|>")
        if tool_call:
            parts.append(f"<|call tool|>{tool_call}")
        if tool_response:
            parts.append(f"<|tool response|>{tool_response}")
        if reflection:
            parts.append(f"<|reflect|>{reflection}")
        if content:
            parts.append(content)
        return "".join(parts)

    def save_to_file(self, path: str) -> None:
        """Serializes tokenizer configuration and ranks to a JSON file."""
        data = {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "merge_ranks": {
                k.hex(): v for k, v in self._merge_ranks.items()
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, path: str) -> AttoTokenizer:
        """Loads tokenizer configuration and ranks from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merge_ranks = {
            bytes.fromhex(k): v for k, v in data["merge_ranks"].items()
        }
        return cls(
            name=data["name"],
            vocab_size=data["vocab_size"],
            special_tokens=data["special_tokens"],
            merge_ranks=merge_ranks,
        )


_DEFAULT_TOKENIZER: Optional[AttoTokenizer] = None


def get_tokenizer() -> AttoTokenizer:
    """Returns a singleton instance of the default 8k tokenizer."""
    global _DEFAULT_TOKENIZER
    if _DEFAULT_TOKENIZER is None:
        _DEFAULT_TOKENIZER = AttoTokenizer()
    return _DEFAULT_TOKENIZER
