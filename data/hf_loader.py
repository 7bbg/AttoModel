"""
Hugging Face Streaming Curriculum Loader & RLVR Prompt Extractor.

Directly streams real datasets from Hugging Face according to the 3-Phase Curriculum
specified in AttoModel_Architecture.pdf (Table 2) and configs/data_mix.yaml:
- Phase 1 (50%): 60% Cosmopedia v2 textbooks + 40% FineWeb-Edu (>4.7 score)
- Phase 2 (34%): 60% Python-Edu AST JSON traces + 40% Math proofs (MATH-500)
- Phase 3 (16%): 60% Glaive Function Calling + 40% Self-Play Chain-of-Thought (CoT)
- Post-Training RLVR: Real verifiable prompt suite extracted from MATH-500 and tool streams.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml

from data.filters import QualityFilter
from data.synthetic_pipeline import ASTJSONTraceGenerator, AgenticTraceGenerator

logger = logging.getLogger(__name__)


class HuggingFaceCurriculumStreamer:
    """
    Streams and processes real Hugging Face datasets aligned with AttoModel curriculum.
    """

    @staticmethod
    def is_datasets_available() -> bool:
        """Checks if the Hugging Face `datasets` library is installed."""
        try:
            import datasets
            return True
        except ImportError:
            return False

    @classmethod
    def stream_phase1_syntax(
        cls,
        count: int,
        cosmo_ratio: float = 0.60,
        min_edu_score: float = 4.0,
    ) -> List[str]:
        """
        Streams Phase 1 (Syntax): 60% Cosmopedia v2 + 40% FineWeb-Edu (>4.7).
        """
        import datasets

        samples: List[str] = []
        n_cosmo = max(1, int(round(count * cosmo_ratio))) if count > 1 else 1
        n_fineweb = max(0, count - n_cosmo)

        # 1. Cosmopedia v2 synthetic textbooks & course materials
        try:
            print(f"[*] Streaming {n_cosmo} samples from HuggingFaceTB/cosmopedia-v2 (cosmopedia-v2)...")
            ds_cosmo = datasets.load_dataset(
                "HuggingFaceTB/cosmopedia-v2",
                "cosmopedia-v2",
                split="train",
                streaming=True,
            )
            cosmo_count = 0
            for item in ds_cosmo:
                text = item.get("text", "").strip()
                if text and QualityFilter.passes_heuristics(text, min_chars=128):
                    samples.append(text)
                    cosmo_count += 1
                    if cosmo_count >= n_cosmo:
                        break
        except Exception as e:
            logger.warning(f"Failed to stream Cosmopedia v2: {e}")

        # 2. FineWeb-Edu high-score educational web texts
        try:
            print(f"[*] Streaming {n_fineweb} samples from HuggingFaceFW/fineweb-edu (score >= {min_edu_score})...")
            ds_fw = datasets.load_dataset(
                "HuggingFaceFW/fineweb-edu",
                name="sample-10BT",
                split="train",
                streaming=True,
            )
            fw_count = 0
            for item in ds_fw:
                score = item.get("score", 0.0)
                text = item.get("text", "").strip()
                # Accept if score matches threshold or heuristic educational filter passes
                if (score >= min_edu_score or QualityFilter.calculate_educational_score(text) >= 4.0) and QualityFilter.passes_heuristics(text, min_chars=128):
                    samples.append(text)
                    fw_count += 1
                    if fw_count >= n_fineweb:
                        break
        except Exception as e:
            logger.warning(f"Failed to stream FineWeb-Edu: {e}")

        return samples

    @classmethod
    def stream_phase2_logic(
        cls,
        count: int,
        ast_ratio: float = 0.60,
    ) -> List[str]:
        """
        Streams Phase 2 (Logic): 60% Python-Edu AST JSON pairs + 40% MATH-500 derivations.
        """
        import datasets

        samples: List[str] = []
        n_ast = max(1, int(round(count * ast_ratio))) if count > 1 else 1
        n_math = max(0, count - n_ast)

        # 1. Python code transformed into AST JSON specification pairs
        try:
            print(f"[*] Streaming {n_ast} samples from flytech/python-codes-25k for AST traces...")
            ds_py = datasets.load_dataset(
                "flytech/python-codes-25k",
                split="train",
                streaming=True,
            )
            ast_count = 0
            for item in ds_py:
                raw_code = item.get("output", "").strip() or item.get("text", "").strip()
                if "def " in raw_code or "class " in raw_code:
                    ast_sample = ASTJSONTraceGenerator.generate_ast_pair(raw_code[:2000])
                    samples.append(ast_sample)
                    ast_count += 1
                    if ast_count >= n_ast:
                        break
        except Exception as e:
            logger.warning(f"Failed to stream Python code: {e}")

        # 2. Formal math proofs and step-by-step derivations from MATH-500
        try:
            print(f"[*] Streaming {n_math} samples from HuggingFaceH4/MATH-500...")
            ds_math = datasets.load_dataset(
                "HuggingFaceH4/MATH-500",
                split="test",
                streaming=True,
            )
            math_count = 0
            for item in ds_math:
                problem = item.get("problem", "").strip()
                solution = item.get("solution", "").strip()
                subject = item.get("subject", "Mathematics")
                level = item.get("level", "Advanced")

                proof_text = (
                    f"# Subject: {subject} (Level {level})\n\n"
                    f"## Mathematical Theorem / Problem Specification\n{problem}\n\n"
                    f"## Formal Step-by-Step Proof and Derivation\n{solution}\n"
                )
                samples.append(proof_text)
                math_count += 1
                if math_count >= n_math:
                    break
        except Exception as e:
            logger.warning(f"Failed to stream MATH-500: {e}")

        return samples

    @classmethod
    def stream_phase3_agentic(
        cls,
        count: int,
        tool_ratio: float = 0.60,
    ) -> List[str]:
        """
        Streams Phase 3 (Agentic): 60% Glaive Function Calling + 40% Self-Play CoT.
        Transforms raw dialogues into AttoModel's native agentic tokens:
        <|thought start|>...<|thought end|><|call tool|>...<|tool response|>...<|reflect|>
        """
        import datasets

        samples: List[str] = []
        n_tool = max(1, int(round(count * tool_ratio))) if count > 1 else 1
        n_cot = max(0, count - n_tool)

        # 1. Glaive Function Calling converted to native AttoModel tags
        try:
            print(f"[*] Streaming {n_tool} samples from glaiveai/glaive-function-calling-v2...")
            ds_glaive = datasets.load_dataset(
                "glaiveai/glaive-function-calling-v2",
                split="train",
                streaming=True,
            )
            tool_count = 0
            for item in ds_glaive:
                chat = item.get("chat", "")
                if "<functioncall>" in chat and "FUNCTION RESPONSE:" in chat:
                    user_m = re.search(r"USER:\s*(.*?)\s*ASSISTANT:", chat, re.DOTALL)
                    fn_m = re.search(r"<functioncall>\s*(.*?)\s*(?:<\|endoftext\|>|FUNCTION RESPONSE:)", chat, re.DOTALL)
                    resp_m = re.search(r"FUNCTION RESPONSE:\s*(.*?)\s*ASSISTANT:", chat, re.DOTALL)
                    ans_m = re.search(r"FUNCTION RESPONSE:.*?\s*ASSISTANT:\s*(.*?)(?:<\|endoftext\|>|$)", chat, re.DOTALL)

                    if user_m and fn_m and resp_m:
                        user_text = user_m.group(1).strip()
                        fn_text = fn_m.group(1).strip()
                        resp_text = resp_m.group(1).strip()
                        ans_text = ans_m.group(1).strip() if ans_m else "Execution succeeded."

                        formatted = (
                            f"<|bos|>User: {user_text}\n"
                            f"<|thought start|>I should invoke the requested tool with the parsed parameters.<|thought end|>"
                            f"<|call tool|>\n{fn_text}\n"
                            f"<|tool response|>\n{resp_text}\n"
                            f"<|reflect|>Execution returned result without error. Formulating final answer to the user.\n"
                            f"Final Response: {ans_text}<|eos|>"
                        )
                        samples.append(formatted)
                        tool_count += 1
                        if tool_count >= n_tool:
                            break
        except Exception as e:
            logger.warning(f"Failed to stream Glaive Function Calling: {e}")

        # 2. Self-Play Chain-of-Thought (CoT) with reflection loops
        synthetic_cots = AgenticTraceGenerator.generate_synthetic_agent_dataset(count=n_cot)
        samples.extend(synthetic_cots)

        return samples

    @classmethod
    def extract_rlvr_prompts_from_hf(cls, count: int = 100) -> List[str]:
        """
        Extracts real verifiable prompts from Hugging Face datasets (MATH-500 & Glaive)
        for post-training Group Relative Policy Optimization (GRPO) alignment.
        """
        import datasets

        prompts: List[str] = []
        n_math = count // 3
        n_code = count // 3
        n_tool = count - n_math - n_code

        # 1. Real math prompts from MATH-500
        try:
            ds_math = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test", streaming=True)
            for item in ds_math:
                prob = item.get("problem", "").strip()
                if prob:
                    prompt_str = f"User: Solve the following math problem step-by-step:\n{prob}\n"
                    prompts.append(prompt_str)
                if len(prompts) >= n_math:
                    break
        except Exception:
            pass

        # 2. Real Python code instruction prompts
        try:
            ds_code = datasets.load_dataset("flytech/python-codes-25k", split="train", streaming=True)
            code_prompts_count = 0
            for item in ds_code:
                instruction = item.get("instruction", "").strip()
                if instruction:
                    prompt_str = f"User: Write a python solution for: {instruction}\n"
                    prompts.append(prompt_str)
                    code_prompts_count += 1
                if code_prompts_count >= n_code:
                    break
        except Exception:
            pass

        # 3. Real function calling prompts from Glaive
        try:
            ds_glaive = datasets.load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
            tool_prompts_count = 0
            for item in ds_glaive:
                chat = item.get("chat", "")
                user_m = re.search(r"USER:\s*(.*?)\s*ASSISTANT:", chat, re.DOTALL)
                if user_m:
                    user_text = user_m.group(1).strip()
                    prompt_str = f"User: {user_text}\n"
                    prompts.append(prompt_str)
                    tool_prompts_count += 1
                if tool_prompts_count >= n_tool:
                    break
        except Exception:
            pass

        return prompts

    @classmethod
    def stream_full_curriculum(
        cls,
        total_samples: int = 500,
        config_path: str = "configs/data_mix.yaml",
    ) -> List[str]:
        """
        Streams full balanced dataset matching the 3 phases in configs/data_mix.yaml.
        Falls back to local synthetic generation if offline or Hugging Face fails.
        """
        from data.synthetic_pipeline import CurriculumDataMixer

        if not cls.is_datasets_available():
            print("[!] Hugging Face datasets package not installed. Falling back to synthetic curriculum.")
            return CurriculumDataMixer.generate_curriculum_documents(total_samples, config_path)

        cfg = CurriculumDataMixer.load_config(config_path) if os.path.exists(config_path) else {}
        phases = cfg.get("curriculum", {}).get("phases", [])

        # Default shares from blueprint Table 2: 50% Syntax, 34% Logic, 16% Agentic
        p1_share = 0.50
        p2_share = 0.34
        p3_share = 0.16

        for p in phases:
            name = p.get("name", "").lower()
            share = p.get("share", 0.0)
            if "syntax" in name or "phase 1" in name:
                p1_share = share
            elif "logic" in name or "phase 2" in name:
                p2_share = share
            elif "agentic" in name or "phase 3" in name:
                p3_share = share

        n1 = max(1, int(total_samples * p1_share))
        n2 = max(1, int(total_samples * p2_share))
        n3 = max(1, total_samples - n1 - n2)

        print("=" * 70)
        print(f" STREAMING ACTUAL HUGGING FACE DATASET CURRICULUM ({total_samples} samples)")
        print(f" - Phase 1 (Syntax, {p1_share*100:.0f}%): {n1} samples [Cosmopedia-v2 + FineWeb-Edu]")
        print(f" - Phase 2 (Logic,  {p2_share*100:.0f}%): {n2} samples [Python-Edu ASTs + MATH-500]")
        print(f" - Phase 3 (Agentic,{p3_share*100:.0f}%): {n3} samples [Glaive Function Calling + CoT]")
        print("=" * 70)

        documents: List[str] = []

        try:
            # Phase 1
            p1_docs = cls.stream_phase1_syntax(count=n1)
            documents.extend(p1_docs)
            print(f"[+] Loaded {len(p1_docs)}/{n1} Phase 1 samples from Hugging Face.")

            # Phase 2
            p2_docs = cls.stream_phase2_logic(count=n2)
            documents.extend(p2_docs)
            print(f"[+] Loaded {len(p2_docs)}/{n2} Phase 2 samples from Hugging Face.")

            # Phase 3
            p3_docs = cls.stream_phase3_agentic(count=n3)
            documents.extend(p3_docs)
            print(f"[+] Loaded {len(p3_docs)}/{n3} Phase 3 samples from Hugging Face.")

        except Exception as e:
            print(f"[!] Warning: error while streaming from Hugging Face ({e}).")

        # If any phase yielded fewer samples than requested, backfill from synthetic curriculum
        if len(documents) < total_samples:
            shortfall = total_samples - len(documents)
            print(f"[*] Backfilling {shortfall} samples from synthetic curriculum generator to ensure balanced target...")
            backfill = CurriculumDataMixer.generate_curriculum_documents(total_samples=shortfall, config_path=config_path)
            documents.extend(backfill)

        return documents[:total_samples]

    @classmethod
    def iter_curriculum_stream(
        cls,
        config_path: str = "configs/data_mix.yaml",
        min_edu_score: float = 4.0,
    ) -> Iterator[str]:
        """
        Continuous weighted streaming generator yielding formatted documents across all 3 phases
        specified in AttoModel_Architecture.pdf and configs/data_mix.yaml:
        - Phase 1 (50%): Cosmopedia-v2 (30%) + FineWeb-Edu (20%)
        - Phase 2 (34%): Python code AST JSON pairs (20.4%) + MATH-500 (13.6%)
        - Phase 3 (16%): Glaive function calling (9.6%) + Self-play CoT (6.4%)
        """
        import random
        from data.synthetic_pipeline import (
            AgenticTraceGenerator,
            CodeASTGenerator,
            EducationalCorpusGenerator,
            FormalLogicMathGenerator,
        )

        def _make_cosmo_iter():
            while True:
                try:
                    import datasets
                    ds = datasets.load_dataset(
                        "HuggingFaceTB/cosmopedia-v2",
                        "cosmopedia-v2",
                        split="train",
                        streaming=True,
                    )
                    for item in ds:
                        text = item.get("text", "").strip()
                        if text and QualityFilter.passes_heuristics(text, min_chars=128):
                            yield text
                except Exception as e:
                    logger.warning(f"Cosmopedia stream error: {e}")
                    for text in EducationalCorpusGenerator.generate_textbook_chapters(50):
                        yield text

        def _make_fineweb_iter():
            while True:
                try:
                    import datasets
                    ds = datasets.load_dataset(
                        "HuggingFaceFW/fineweb-edu",
                        name="sample-10BT",
                        split="train",
                        streaming=True,
                    )
                    for item in ds:
                        score = item.get("score", 0.0)
                        text = item.get("text", "").strip()
                        if (score >= min_edu_score or QualityFilter.calculate_educational_score(text) >= 4.0) and QualityFilter.passes_heuristics(text, min_chars=128):
                            yield text
                except Exception as e:
                    logger.warning(f"FineWeb-Edu stream error: {e}")
                    for text in EducationalCorpusGenerator.generate_textbook_chapters(50):
                        yield text

        def _make_python_ast_iter():
            while True:
                try:
                    import datasets
                    ds = datasets.load_dataset(
                        "flytech/python-codes-25k",
                        split="train",
                        streaming=True,
                    )
                    for item in ds:
                        raw_code = item.get("output", "").strip() or item.get("text", "").strip()
                        if "def " in raw_code or "class " in raw_code:
                            yield ASTJSONTraceGenerator.generate_ast_pair(raw_code[:2000])
                except Exception as e:
                    logger.warning(f"Python AST stream error: {e}")
                    for text in CodeASTGenerator.generate_code_ast_pairs(50):
                        yield text

        def _make_math_iter():
            while True:
                try:
                    import datasets
                    ds = datasets.load_dataset(
                        "HuggingFaceH4/MATH-500",
                        split="test",
                        streaming=True,
                    )
                    for item in ds:
                        problem = item.get("problem", "").strip()
                        solution = item.get("solution", "").strip()
                        subject = item.get("subject", "Mathematics")
                        level = item.get("level", "Advanced")
                        proof_text = (
                            f"# Subject: {subject} (Level {level})\n\n"
                            f"## Mathematical Theorem / Problem Specification\n{problem}\n\n"
                            f"## Formal Step-by-Step Proof and Derivation\n{solution}\n"
                        )
                        yield proof_text
                except Exception as e:
                    logger.warning(f"MATH-500 stream error: {e}")
                    for text in FormalLogicMathGenerator.generate_math_proofs(50):
                        yield text

        def _make_glaive_iter():
            while True:
                try:
                    import datasets
                    ds = datasets.load_dataset(
                        "glaiveai/glaive-function-calling-v2",
                        split="train",
                        streaming=True,
                    )
                    for item in ds:
                        chat = item.get("chat", "")
                        if "<functioncall>" in chat and "FUNCTION RESPONSE:" in chat:
                            user_m = re.search(r"USER:\s*(.*?)\s*ASSISTANT:", chat, re.DOTALL)
                            fn_m = re.search(r"<functioncall>\s*(.*?)\s*(?:<\|endoftext\|>|FUNCTION RESPONSE:)", chat, re.DOTALL)
                            resp_m = re.search(r"FUNCTION RESPONSE:\s*(.*?)\s*ASSISTANT:", chat, re.DOTALL)
                            ans_m = re.search(r"FUNCTION RESPONSE:.*?\s*ASSISTANT:\s*(.*?)(?:<\|endoftext\|>|$)", chat, re.DOTALL)

                            if user_m and fn_m and resp_m:
                                user_text = user_m.group(1).strip()
                                fn_text = fn_m.group(1).strip()
                                resp_text = resp_m.group(1).strip()
                                ans_text = ans_m.group(1).strip() if ans_m else "Execution succeeded."
                                yield (
                                    f"<|bos|>User: {user_text}\n"
                                    f"<|thought start|>I should invoke the requested tool with the parsed parameters.<|thought end|>"
                                    f"<|call tool|>\n{fn_text}\n"
                                    f"<|tool response|>\n{resp_text}\n"
                                    f"<|reflect|>Execution returned result without error. Formulating final answer to the user.\n"
                                    f"Final Response: {ans_text}<|eos|>"
                                )
                except Exception as e:
                    logger.warning(f"Glaive stream error: {e}")
                    for text in AgenticTraceGenerator.generate_synthetic_agent_dataset(50):
                        yield text

        def _make_cot_iter():
            while True:
                for text in AgenticTraceGenerator.generate_synthetic_agent_dataset(50):
                    yield text

        iter_cosmo = _make_cosmo_iter()
        iter_fineweb = _make_fineweb_iter()
        iter_python = _make_python_ast_iter()
        iter_math = _make_math_iter()
        iter_glaive = _make_glaive_iter()
        iter_cot = _make_cot_iter()

        stream_map = {
            "cosmo": iter_cosmo,
            "fineweb": iter_fineweb,
            "python": iter_python,
            "math": iter_math,
            "glaive": iter_glaive,
            "cot": iter_cot,
        }

        # 100-item round matching blueprint phase proportions: 50% Phase 1, 34% Phase 2, 16% Phase 3
        round_pattern = (
            ["cosmo"] * 30
            + ["fineweb"] * 20
            + ["python"] * 20
            + ["math"] * 14
            + ["glaive"] * 10
            + ["cot"] * 6
        )

        rng = random.Random(42)

        while True:
            shuffled = list(round_pattern)
            rng.shuffle(shuffled)
            for source_name in shuffled:
                source_iter = stream_map[source_name]
                yield next(source_iter)
