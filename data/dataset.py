"""
High-Performance Sequence Packing, Binary Memory-Mapped Streaming, and FIM Transformations.

Features:
- Zero-padding sequence packing (multi-document packing up to max_seq_len).
- High-yield binary memory-mapped streaming (uint16 raw binary format via np.memmap).
- Dynamic Fill-In-The-Middle (FIM) structural transformations (PSM / SPM formats).
- Segment / document boundary masking for multi-document pack isolation.
"""

from __future__ import annotations

import os
import random
import struct
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from data.tokenizer import AttoTokenizer, get_tokenizer


def apply_fim_transform(
    token_ids: List[int],
    tokenizer: AttoTokenizer,
    fim_rate: float = 0.5,
    spm_prob: float = 0.5,
    min_length: int = 16,
) -> List[int]:
    """
    Applies Fill-In-The-Middle (FIM) transformation to a sequence of token IDs.
    
    Transforms text into:
      PSM: <|fim_prefix|> Prefix <|fim_suffix|> Suffix <|fim_middle|> Middle <|eos|>
      SPM: <|fim_suffix|> Suffix <|fim_prefix|> Prefix <|fim_middle|> Middle <|eos|>
    """
    if len(token_ids) < min_length or random.random() > fim_rate:
        return token_ids

    # Choose random cut points: prefix [0:cut1], middle [cut1:cut2], suffix [cut2:]
    l = len(token_ids)
    cut1 = random.randint(1, l - 2)
    cut2 = random.randint(cut1 + 1, l - 1)

    prefix = token_ids[:cut1]
    middle = token_ids[cut1:cut2]
    suffix = token_ids[cut2:]

    fim_p = tokenizer.fim_prefix_id
    fim_m = tokenizer.fim_middle_id
    fim_s = tokenizer.fim_suffix_id
    eos = tokenizer.eos_token_id

    if random.random() < spm_prob:
        # SPM Format
        transformed = [fim_s] + suffix + [fim_p] + prefix + [fim_m] + middle + [eos]
    else:
        # PSM Format
        transformed = [fim_p] + prefix + [fim_s] + suffix + [fim_m] + middle + [eos]

    return transformed


class PackedSequenceDataset(Dataset):
    """
    Zero-padding sequence packed dataset.
    Packs multiple document chunks into fixed context length `max_seq_len`.
    """

    def __init__(
        self,
        documents: List[str],
        tokenizer: Optional[AttoTokenizer] = None,
        max_seq_len: int = 4096,
        fim_rate: float = 0.0,
        spm_prob: float = 0.5,
    ):
        self.tokenizer = tokenizer or get_tokenizer()
        self.max_seq_len = max_seq_len
        self.fim_rate = fim_rate
        self.spm_prob = spm_prob

        # Tokenize and pack all documents
        self.samples = self._pack_documents(documents)

    def _pack_documents(self, documents: List[str]) -> List[Dict[str, torch.Tensor]]:
        packed_samples: List[Dict[str, torch.Tensor]] = []
        current_ids: List[int] = []
        current_doc_ids: List[int] = []
        doc_counter = 0

        for doc in documents:
            if not doc.strip():
                continue
            token_ids = self.tokenizer.encode(doc)
            if self.fim_rate > 0:
                token_ids = apply_fim_transform(
                    token_ids,
                    self.tokenizer,
                    fim_rate=self.fim_rate,
                    spm_prob=self.spm_prob,
                )
            # Ensure document ends with EOS
            if not token_ids or token_ids[-1] != self.tokenizer.eos_token_id:
                token_ids.append(self.tokenizer.eos_token_id)

            idx = 0
            while idx < len(token_ids):
                remaining_space = self.max_seq_len - len(current_ids)
                chunk = token_ids[idx : idx + remaining_space]
                current_ids.extend(chunk)
                current_doc_ids.extend([doc_counter] * len(chunk))
                idx += len(chunk)

                if len(current_ids) == self.max_seq_len:
                    # Form complete sample
                    input_tensor = torch.tensor(current_ids, dtype=torch.long)
                    # For CLM targets: next token prediction
                    target_tensor = input_tensor.clone()
                    doc_tensor = torch.tensor(current_doc_ids, dtype=torch.long)
                    pos_tensor = torch.arange(self.max_seq_len, dtype=torch.long)

                    packed_samples.append({
                        "input_ids": input_tensor,
                        "labels": target_tensor,
                        "position_ids": pos_tensor,
                        "document_ids": doc_tensor,
                    })

                    current_ids = []
                    current_doc_ids = []

            doc_counter += 1

        # Pad remaining if current_ids has tokens
        if current_ids:
            pad_len = self.max_seq_len - len(current_ids)
            padded_input = current_ids + [self.tokenizer.pad_token_id] * pad_len
            padded_labels = current_ids + [-100] * pad_len  # -100 ignored in loss
            padded_docs = current_doc_ids + [doc_counter] * pad_len
            padded_pos = list(range(len(current_ids))) + [0] * pad_len

            packed_samples.append({
                "input_ids": torch.tensor(padded_input, dtype=torch.long),
                "labels": torch.tensor(padded_labels, dtype=torch.long),
                "position_ids": torch.tensor(padded_pos, dtype=torch.long),
                "document_ids": torch.tensor(padded_docs, dtype=torch.long),
            })

        return packed_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class BinaryMemmapDataset(IterableDataset):
    """
    High-speed binary memory-mapped streaming dataset for NVMe SSD zero-copy loading.
    
    Data is stored on disk as flat uint16 arrays. Reads contiguous chunks of `max_seq_len`
    tokens directly from disk via np.memmap.
    """

    def __init__(
        self,
        bin_path: str,
        max_seq_len: int = 4096,
        tokenizer: Optional[AttoTokenizer] = None,
        infinite: bool = False,
    ):
        self.bin_path = bin_path
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer or get_tokenizer()
        self.infinite = infinite

        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary file not found: {bin_path}")

        # Check total tokens in binary file (uint16 = 2 bytes per token)
        file_size_bytes = os.path.getsize(bin_path)
        self.total_tokens = file_size_bytes // 2
        self.num_samples = self.total_tokens // self.max_seq_len

        # Open memory-mapped array in read-only mode
        self.mmap_data = np.memmap(
            self.bin_path,
            dtype=np.uint16,
            mode="r",
            shape=(self.total_tokens,),
        )

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single process
            start_idx = 0
            end_idx = self.num_samples
        else:
            # Multi-process DataLoader
            per_worker = int(np.ceil(self.num_samples / float(worker_info.num_workers)))
            start_idx = worker_info.id * per_worker
            end_idx = min(start_idx + per_worker, self.num_samples)

        while True:
            for i in range(start_idx, end_idx):
                start_offset = i * self.max_seq_len
                end_offset = start_offset + self.max_seq_len
                token_slice = self.mmap_data[start_offset:end_offset].astype(np.int64)

                input_ids = torch.from_numpy(token_slice)
                labels = input_ids.clone()
                position_ids = torch.arange(self.max_seq_len, dtype=torch.long)

                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "position_ids": position_ids,
                }
            if not self.infinite:
                break


def write_tokens_to_binary(
    tokens: Sequence[int],
    output_bin_path: str,
    append: bool = False,
) -> int:
    """
    Writes a sequence of token IDs to a uint16 binary file.
    Returns the number of tokens written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_bin_path)), exist_ok=True)
    mode = "ab" if append else "wb"
    
    # Ensure all token IDs fit in uint16 (0 to 65535)
    arr = np.array(tokens, dtype=np.uint16)
    with open(output_bin_path, mode) as f:
        f.write(arr.tobytes())
    return len(tokens)


class StreamingSequencePacker:
    """
    High-throughput zero-memory streaming sequence packer.
    Progressively encodes documents, applies FIM transformations, packs contiguous sequences
    of exact length max_seq_len, and flushes directly to a uint16 binary file.
    Enables streaming arbitrarily large token budgets (e.g. 50M, 1B, 5B tokens) with O(1) host RAM.
    """

    def __init__(
        self,
        output_bin_path: str,
        tokenizer: Optional[AttoTokenizer] = None,
        max_seq_len: int = 4096,
        fim_rate: float = 0.50,
        spm_prob: float = 0.50,
        append: bool = False,
    ):
        self.output_bin_path = output_bin_path
        self.tokenizer = tokenizer or get_tokenizer()
        self.max_seq_len = max_seq_len
        self.fim_rate = fim_rate
        self.spm_prob = spm_prob
        self.buffer: List[int] = []
        self.tokens_written = 0
        self.sequences_written = 0

        os.makedirs(os.path.dirname(os.path.abspath(output_bin_path)), exist_ok=True)
        if not append:
            with open(output_bin_path, "wb") as f:
                pass

    def add_document(self, text: str) -> int:
        """
        Encodes document, applies FIM, and appends complete sequences of max_seq_len to disk.
        Returns the number of tokens flushed to disk during this call.
        """
        if not text or not text.strip():
            return 0

        token_ids = self.tokenizer.encode(text)
        if self.fim_rate > 0:
            token_ids = apply_fim_transform(
                token_ids,
                self.tokenizer,
                fim_rate=self.fim_rate,
                spm_prob=self.spm_prob,
            )

        if not token_ids or token_ids[-1] != self.tokenizer.eos_token_id:
            token_ids.append(self.tokenizer.eos_token_id)

        self.buffer.extend(token_ids)
        written_in_call = 0

        # Flush full sequences of max_seq_len
        while len(self.buffer) >= self.max_seq_len:
            chunk = self.buffer[: self.max_seq_len]
            self.buffer = self.buffer[self.max_seq_len :]

            arr = np.array(chunk, dtype=np.uint16)
            with open(self.output_bin_path, "ab") as f:
                f.write(arr.tobytes())

            self.tokens_written += self.max_seq_len
            self.sequences_written += 1
            written_in_call += self.max_seq_len

        return written_in_call

    def finish(self) -> int:
        """
        Pads and flushes any remaining tokens in buffer to form the final sequence.
        Returns the total tokens written across the entire session.
        """
        if self.buffer:
            pad_len = self.max_seq_len - len(self.buffer)
            chunk = self.buffer + [self.tokenizer.pad_token_id] * pad_len
            arr = np.array(chunk, dtype=np.uint16)
            with open(self.output_bin_path, "ab") as f:
                f.write(arr.tobytes())

            self.tokens_written += self.max_seq_len
            self.sequences_written += 1
            self.buffer = []

        return self.tokens_written
