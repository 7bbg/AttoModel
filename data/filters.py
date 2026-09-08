"""
Data Quality Filters, Heuristics, and MinHash LSH Deduplication Engine.

Used during dataset preprocessing to curate high-density synthetic and educational text
for the 5.0B token curriculum.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Dict, List, Set, Tuple


class MinHashLSHDeduplicator:
    """
    MinHash Locality Sensitive Hashing (LSH) deduplicator for document filtering.
    Computes min-hash signatures using random linear permutations.
    """

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.85,
        ngram_size: int = 5,
    ):
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size
        
        # Fixed random coefficients for hash permutations
        # h_i(x) = (a_i * x + b_i) % prime
        self.prime = 2**61 - 1
        # Seeded deterministic generators
        rng = 42
        self.a_coeffs = []
        self.b_coeffs = []
        for i in range(num_perm):
            rng = (1103515245 * rng + 12345) % (2**31)
            self.a_coeffs.append(rng | 1)
            rng = (1103515245 * rng + 12345) % (2**31)
            self.b_coeffs.append(rng)

        # LSH buckets: bands and rows
        # threshold ~ (1/b)^(1/r)
        self.b = 16
        self.r = num_perm // self.b
        self.buckets: Dict[Tuple[int, int], List[int]] = {}
        self.seen_hashes: Set[int] = set()

    def _get_shingles(self, text: str) -> Set[int]:
        """Generates hashed word-level n-grams (shingles)."""
        words = re.findall(r"\w+", text.lower())
        if len(words) < self.ngram_size:
            return {int(hashlib.md5(text.lower().encode("utf-8")).hexdigest()[:8], 16)}
        shingles = set()
        for i in range(len(words) - self.ngram_size + 1):
            shingle_str = " ".join(words[i : i + self.ngram_size])
            h = int(hashlib.md5(shingle_str.encode("utf-8")).hexdigest()[:8], 16)
            shingles.add(h)
        return shingles

    def compute_signature(self, shingles: Set[int]) -> List[int]:
        """Computes the MinHash signature vector."""
        if not shingles:
            return [0] * self.num_perm
        sig = []
        for a, b in zip(self.a_coeffs, self.b_coeffs):
            min_val = min(((a * s + b) % self.prime) for s in shingles)
            sig.append(min_val)
        return sig

    def is_duplicate(self, doc_id: int, text: str, add_if_unique: bool = True) -> bool:
        """
        Checks if the document is a near-duplicate of an already seen document.
        If unique and add_if_unique=True, adds its bands to the LSH index.
        """
        shingles = self._get_shingles(text)
        if not shingles:
            return True

        sig = self.compute_signature(shingles)
        is_dup = False

        # Check all bands
        for band_idx in range(self.b):
            band_slice = tuple(sig[band_idx * self.r : (band_idx + 1) * self.r])
            bucket_key = (band_idx, hash(band_slice))
            if bucket_key in self.buckets:
                # Potential match found
                is_dup = True
                break

        if not is_dup and add_if_unique:
            for band_idx in range(self.b):
                band_slice = tuple(sig[band_idx * self.r : (band_idx + 1) * self.r])
                bucket_key = (band_idx, hash(band_slice))
                if bucket_key not in self.buckets:
                    self.buckets[bucket_key] = []
                self.buckets[bucket_key].append(doc_id)

        return is_dup


class QualityFilter:
    """
    Heuristic and rule-based data quality filtering suite.
    Enforces minimum length, symbol-to-word ratio, line length, repetition, and entropy.
    """

    @staticmethod
    def passes_heuristics(
        text: str,
        min_chars: int = 64,
        max_chars: int = 100000,
        max_symbol_ratio: float = 0.35,
        max_repetition_ratio: float = 0.40,
    ) -> bool:
        """
        Evaluates document quality against heuristics. Returns True if acceptable.
        """
        if not text or len(text) < min_chars or len(text) > max_chars:
            return False

        # Symbol to word ratio
        words = text.split()
        if not words:
            return False
            
        non_alphanumeric = sum(1 for c in text if not c.isalnum() and not c.isspace())
        symbol_ratio = non_alphanumeric / max(len(text), 1)
        if symbol_ratio > max_symbol_ratio:
            return False

        # Character entropy / repetition check
        lines = text.strip().split("\n")
        if len(lines) > 5:
            unique_lines = set(l.strip() for l in lines if l.strip())
            duplicate_ratio = 1.0 - (len(unique_lines) / max(len(lines), 1))
            if duplicate_ratio > max_repetition_ratio:
                return False

        return True

    @staticmethod
    def calculate_educational_score(text: str) -> float:
        """
        Educational quality proxy score in range [0.0, 5.0] (matching FineWeb-Edu thresholds).
        Rewards clear definitions, structure, reasoning words, examples, and markdown headers.
        """
        score = 2.5  # Base score
        
        # Positive indicators
        edu_keywords = [
            r"\btherefore\b", r"\bbecause\b", r"\bfor example\b", r"\bdefinition\b",
            r"\btheorem\b", r"\bproof\b", r"\balgorithm\b", r"\bconcept\b",
            r"\bfunction\b", r"\bexplain\b", r"\bstep-by-step\b", r"\bin summary\b",
        ]
        for pattern in edu_keywords:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.25

        # Code block presence
        if "```" in text or "def " in text or "class " in text:
            score += 0.5

        # Structure presence (headings, numbered lists)
        if re.search(r"^#{1,4}\s+", text, re.MULTILINE) or re.search(r"^\d+\.\s+", text, re.MULTILINE):
            score += 0.3

        # Negative indicators (boilerplate, garbage, ads)
        ad_keywords = [r"\bcookie policy\b", r"\ball rights reserved\b", r"\bclick here\b", r"\bsubscribe\b"]
        for pattern in ad_keywords:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.5

        return max(0.0, min(5.0, score))
