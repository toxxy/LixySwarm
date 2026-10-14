"""Reproducible, public-only instruction gate for a conversational seed.

This small gate checks elementary behaviour. Passing it is not evidence of
distributed learning, broad reasoning quality, or the usefulness of the swarm.
Models and runtime outputs belong outside the source tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

MODEL_REPOSITORY = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PINNED_FILES = {
    "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
}
LARGER_MODEL = {
    "repository": "Qwen/Qwen3-1.7B",
    "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "files": {
        "model-00001-of-00002.safetensors": "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
        "model-00002-of-00002.safetensors": "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9",
        "tokenizer.json": PINNED_FILES["tokenizer.json"],
        "config.json": "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197",
        "model.safetensors.index.json": "0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d",
        **{name: PINNED_FILES[name] for name in ("generation_config.json", "tokenizer_config.json", "LICENSE")},
    },
}

# Fixed before running the seed. No personal data, conversation, or training
# corpus is included. Matching rules are intentionally stricter for exact tasks.
CASES = [
    ("es_addition", "Responde únicamente con el resultado: ¿cuánto es 2 + 2?", r"^4[.!]?$"),
    ("en_multiplication", "Reply with just the number: what is 7 times 8?", r"^56[.!]?$"),
    ("es_translation", "Traduce al inglés únicamente esta frase: El gato está en la casa.", r"(?i)^(the )?cat is (in|inside) the house[.!]?$"),
    ("en_translation", "Translate into Spanish, without explanation: Good morning.", r"(?i)^buenos días[.!]?$"),
    ("es_instruction", "Escribe exactamente la palabra conectada y nada más.", r"^conectada[.!]?$"),
    ("en_instruction", "Write exactly the word ready and nothing else.", r"^ready[.!]?$"),
    ("es_fact", "¿Cuál es la capital de Francia? Responde en una oración breve en español.", r"(?i)par[íi]s"),
    ("en_fact", "What is the chemical formula of water? Reply with only the formula.", r"^H[₂2]O[.!]?$"),
    ("es_order", "Ordena estos números de menor a mayor. Responde solo con la lista: 9, 2, 5.", r"^2\s*,\s*5\s*,\s*9[.]?$"),
    ("en_extract", 'Given this JSON: {"node": "ant", "status": "online"}. Return only the value of status.', r"^online[.!]?$"),
    ("es_public_context", "Usa solo estos datos: El VPS de LixySwarm sirve de relay y descubrimiento. No entrena modelos. ¿El VPS entrena? Responde sí o no.", r"(?i)^no[.!]?$"),
    ("en_public_context", "Use these facts: an ant is a physical computer; experts are models hosted on it. Can one ant host two experts? Answer yes or no.", r"(?i)^yes[.!]?$"),
]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--model-size", choices=["0.6b", "1.7b"], default="0.6b")
    parser.add_argument("--decoding", choices=["greedy", "qwen-recommended"], default="greedy")
    options = parser.parse_args()
    repository, revision, pinned = MODEL_REPOSITORY, MODEL_REVISION, PINNED_FILES
    if options.model_size == "1.7b":
        repository, revision, pinned = LARGER_MODEL["repository"], LARGER_MODEL["revision"], LARGER_MODEL["files"]
    files = {}
    for name, expected in pinned.items():
        path = options.model_directory / name
        actual = file_hash(path)
        if actual != expected:
            raise ValueError(f"Pinned public artifact mismatch: {name}")
        files[name] = {"sha256": actual, "bytes": path.stat().st_size}
    print("Pinned artifacts verified", flush=True)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(4)
    torch.manual_seed(20261014)
    if options.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.25)
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(options.model_directory, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        options.model_directory, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16 if options.device == "cuda" else torch.float32,
    ).to(options.device).eval()
    load_seconds = time.monotonic() - started
    results = []
    for case_id, prompt, pattern in CASES:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(encoded, return_tensors="pt").to(options.device)
        started = time.monotonic()
        generation = {"do_sample": False}
        if options.decoding == "qwen-recommended":
            generation = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=96, use_cache=True, **generation)
        tokens = output[0, inputs.input_ids.shape[1]:]
        answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        passed = re.search(pattern, answer) is not None
        results.append({"case": case_id, "prompt": prompt, "answer": answer,
                        "passed": passed, "seconds": time.monotonic() - started,
                        "output_tokens": int(tokens.numel())})
        print(json.dumps({"case": case_id, "passed": passed, "answer": answer}, ensure_ascii=False), flush=True)
    passed = sum(case["passed"] for case in results)
    # At least 11/12, all exact instruction and contextual role
    # checks must pass. These checks are eligibility for further integration.
    required = {"es_instruction", "en_instruction", "es_public_context", "en_public_context"}
    eligible = passed >= 11 and all(case["passed"] for case in results if case["case"] in required)
    report = {"schema_version": 1, "model_repository": repository,
              "model_revision": revision, "license": "Apache-2.0",
              "source": f"https://huggingface.co/{repository}",
              "public_data_only": True, "remote_code": False,
              "torch_version": torch.__version__, "transformers_version": transformers.__version__,
              "device": options.device, "load_seconds": load_seconds,
              "decoding": options.decoding, "random_seed": 20261014,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if options.device == "cuda" else None,
              "gate": {"minimum_passed": 11, "total": 12, "passed": passed,
                       "required_cases": sorted(required), "eligible_for_integration": eligible},
              "files": files, "cases": results,
              "limitations": ["Elementary instruction gate only", "No distributed-learning claim",
                              "No swarm, memory, or specialist benefit measured", "Non-thinking decoding"]}
    options.report.parent.mkdir(parents=True, exist_ok=True)
    options.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["gate"]), flush=True)


if __name__ == "__main__":
    main()
