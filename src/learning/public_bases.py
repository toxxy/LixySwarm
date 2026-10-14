"""Immutable upstream provenance for the two public language research bases."""
import hashlib
import json

SMALL_REPOSITORY="Qwen/Qwen3-0.6B"
SMALL_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"
SMALL_FILES={
    "model.safetensors":"f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    "tokenizer.json":"aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "config.json":"660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "generation_config.json":"2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "tokenizer_config.json":"d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "LICENSE":"832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
}
LARGE_REPOSITORY="Qwen/Qwen3-1.7B"
LARGE_REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
LARGE_FILES={
    "model-00001-of-00002.safetensors":"169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
    "model-00002-of-00002.safetensors":"912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9",
    "tokenizer.json":SMALL_FILES["tokenizer.json"],
    "config.json":"1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197",
    "model.safetensors.index.json":"0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d",
    **{name:SMALL_FILES[name] for name in ("generation_config.json","tokenizer_config.json","LICENSE")},
}
BASES={"qwen3-0.6b":SMALL_FILES,"qwen3-1.7b":LARGE_FILES}


def base_hash(name):
    files=BASES[name]
    return SMALL_FILES["model.safetensors"] if name=="qwen3-0.6b" else hashlib.sha256(json.dumps(files,sort_keys=True,separators=(",",":")).encode()).hexdigest()
