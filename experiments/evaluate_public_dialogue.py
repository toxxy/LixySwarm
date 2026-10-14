"""Fixed broader public/synthetic dialogue assessment after the SFT gate.

These prompts are absent from the training corpus. Keyword checks are a rough
screen and all answers remain available for human quality inspection.
"""
from __future__ import annotations
import argparse
import json
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CASES=[
    ("es_germany","¿Cuál es la capital de Alemania? Contesta brevemente.",r"(?i)berl[íi]n"),
    ("en_italy","What is the capital of Italy? Keep it brief.",r"(?i)rome"),
    ("es_spain","¿Cuál es la capital de España?",r"(?i)madrid"),
    ("en_japan","What is the capital of Japan?",r"(?i)tokyo"),
    ("es_week","¿Cuántos días tiene una semana? Responde con un número.",r"^7[.!]?$"),
    ("en_triangle","How many sides does a triangle have? Reply with a number.",r"^3[.!]?$"),
    ("es_subtraction","Da solo el resultado de 15 menos 6.",r"^9[.!]?$"),
    ("en_division","Reply only with the result of 81 divided by 9.",r"^9[.!]?$"),
    ("es_translate","Traduce al inglés sin explicación: Gracias por ayudarme.",r"(?i)thank.*help"),
    ("en_translate","Translate into Spanish with no explanation: I like learning.",r"(?i)me gusta aprender"),
    ("es_definition","Explica en español y en una oración qué es una red descentralizada.",r"(?i)(nodo|equipo|participante|central)"),
    ("en_definition","Explain a decentralized network in one English sentence.",r"(?i)(node|computer|central|participant)"),
    ("es_replicas","Una red tiene copias del mismo modelo en tres equipos. Si uno se apaga, ¿se pierde necesariamente el modelo? Responde y explica brevemente.",r"(?i)(no|otras copias|otros equipos)"),
    ("en_replicas","Three computers have copies of a model. If one shuts down, is the model necessarily lost? Give a short explanation.",r"(?i)(no|other cop|remaining)"),
    ("es_privacy","Datos: la memoria personal se guarda solo en el equipo del usuario. ¿Debe enviarse automáticamente a otros nodos? Responde sí o no.",r"(?i)^no[.!]?$"),
    ("en_privacy","Facts: personal memory stays on the user's own computer. Should it automatically be sent to other nodes? Answer yes or no.",r"(?i)^no[.!]?$"),
    ("es_role","Datos: Aurora solo transporta mensajes; Boreal calcula gradientes. ¿Cuál calcula gradientes? Escribe solo su nombre.",r"^Boreal[.!]?$"),
    ("en_role","Facts: Aurora only forwards messages; Boreal computes gradients. Which computes gradients? Write only its name.",r"^Boreal[.!]?$"),
    ("es_json","Extrae solo el valor de active del JSON {\"active\": false, \"count\": 12}.",r"(?i)^false[.!]?$"),
    ("en_json","Extract only the value of count from JSON {\"active\": false, \"count\": 12}.",r"^12[.!]?$"),
    ("es_homogeneous","Dos modelos tienen matrices de tamaños diferentes. ¿Es correcto sumar sus gradientes directamente? Responde sí o no.",r"(?i)^no[.!]?$"),
    ("en_homogeneous","Two models have differently shaped parameter matrices. Is it correct to directly add their gradients? Answer yes or no.",r"(?i)^no[.!]?$"),
    ("es_compare","Datos: A procesa 20 fragmentos por minuto y B procesa 8. ¿Cuál procesa más por minuto? Responde A o B.",r"^A[.!]?$"),
    ("en_compare","Facts: A processes 20 shards per minute and B processes 8. Which processes more per minute? Reply A or B.",r"^A[.!]?$"),
]
MULTITURN=[
    ("es_context",[{"role":"user","content":"En este ejemplo ficticio, la clave de la tarea es lavanda."},
                   {"role":"assistant","content":"La clave de esta tarea es lavanda."},
                   {"role":"user","content":"¿Cuál es la clave? Responde solo la palabra."}],r"^lavanda[.!]?$"),
    ("en_context",[{"role":"user","content":"In this fictional example, the task label is amber."},
                   {"role":"assistant","content":"The task label is amber."},
                   {"role":"user","content":"What is the label? Reply with only the word."}],r"^amber[.!]?$"),
    ("es_correction",[{"role":"user","content":"En este ejemplo, el equipo de interés se llama Azul."},
                      {"role":"assistant","content":"El equipo se llama Azul."},
                      {"role":"user","content":"Corrijo el ejemplo: se llama Coral. ¿Cómo se llama ahora? Solo el nombre."}],r"^Coral[.!]?$"),
    ("en_correction",[{"role":"user","content":"The example node is called Blue."},
                      {"role":"assistant","content":"The node is Blue."},
                      {"role":"user","content":"Correction: it is called Coral. What is its name now? Just the name."}],r"^Coral[.!]?$"),
]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--model-directory",type=Path,required=True)
    parser.add_argument("--adapter",type=Path,required=True);parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--model-size",choices=["0.6b","1.7b"],default="0.6b")
    parser.add_argument("--direct-device-load",action="store_true")
    args=parser.parse_args()
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from src.learning.language_adapter import install_adapter,load_adapter,adapters_enabled
    from experiments.evaluate_conversation_seed import PINNED_FILES,LARGER_MODEL,file_hash
    pinned=PINNED_FILES if args.model_size=="0.6b" else LARGER_MODEL["files"]
    base_hash=PINNED_FILES["model.safetensors"] if args.model_size=="0.6b" else hashlib.sha256(json.dumps(pinned,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    from src.learning.public_conversation import public_corpus
    train={row["prompt"] for row in public_corpus()["train"]}
    assert not train&{row[1] for row in CASES}
    for name,expected in pinned.items():
        if file_hash(args.model_directory/name)!=expected:raise ValueError("Pinned base mismatch")
    torch.set_num_threads(4);torch.manual_seed(20261014);torch.cuda.set_per_process_memory_fraction(0.25)
    tokenizer=AutoTokenizer.from_pretrained(args.model_directory,local_files_only=True,trust_remote_code=False)
    loading={"device_map":{"":"cuda"}} if args.direct_device_load else {}
    model=AutoModelForCausalLM.from_pretrained(args.model_directory,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,**loading).to("cuda").eval()
    install_adapter(model);load_adapter(model,args.adapter,base_hash)
    results={}
    for adapted in [False,True]:
        rows=[]
        with adapters_enabled(model,adapted):
            for case,messages,pattern in [(name,[{"role":"user","content":prompt}],pattern) for name,prompt,pattern in CASES]+MULTITURN:
                text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                inputs=tokenizer(text,return_tensors="pt").to("cuda")
                with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=128,do_sample=False,use_cache=True)
                answer=tokenizer.decode(out[0,inputs.input_ids.shape[1]:],skip_special_tokens=True).strip()
                rows.append({"case":case,"messages":messages,"answer":answer,"passed":re.search(pattern,answer) is not None})
                print(json.dumps({"adapter":adapted,"case":case,"passed":rows[-1]["passed"],"answer":answer},ensure_ascii=False),flush=True)
        results["candidate" if adapted else "baseline"]={"passed":sum(row["passed"] for row in rows),"total":len(rows),"cases":rows}
    previously_correct={row["case"] for row in results["baseline"]["cases"] if row["passed"]}
    regressions=[row["case"] for row in results["candidate"]["cases"] if row["case"] in previously_correct and not row["passed"]]
    report={"schema_version":1,"scope":"Broader public/synthetic screening with all outputs available for review",
            "model_size":args.model_size,"base_model_hash":base_hash,
            "adapter_hash":file_hash(args.adapter),"private_conversations_loaded":False,"results":results,"regressions":regressions,
            "limitations":["Keyword checks do not prove explanation correctness or fluent conversation", "Synthetic context only; no persistent Matriarca tested", "No network inference or training in this assessment"]}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"baseline":results["baseline"]["passed"],"candidate":results["candidate"]["passed"],"total":28,"regressions":regressions}),flush=True)


if __name__=="__main__":main()
