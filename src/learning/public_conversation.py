"""Versioned synthetic instruction data, with disjoint train/evaluation inputs.

All text is generated from explicit public literals. This module never opens a
conversation, personal checkpoint, memory bank, browser history, or directory.
"""
from __future__ import annotations
import hashlib
import json
import random

DATASET_VERSION="lixyswarm-public-instructions-v2"


def public_corpus(version=DATASET_VERSION):
    if version not in {"lixyswarm-public-instructions-v1",DATASET_VERSION}:raise ValueError("Unknown public corpus version")
    train=[];evaluation=[]
    def add(target,kind,prompt,answer):target.append({"kind":kind,"prompt":prompt,"answer":answer})
    for target,numbers in [(train,range(11,27)),(evaluation,range(31,35))]:
        for n in numbers:
            for language in ["es","en"]:
                add(target,"arithmetic",f"Da solo el número, sin explicación: {n} + 3." if language=="es" else f"Give just the number, with no explanation: {n} + 3.",str(n+3))
                add(target,"arithmetic",f"Contesta únicamente el producto de {n} por 2." if language=="es" else f"Reply only with the product of {n} and 2.",str(n*2))
                values=[n+5,n,n+2]
                add(target,"sorting",f"Ordena {values} de menor a mayor. Usa números separados por comas sin corchetes." if language=="es" else f"Sort {values} in ascending order. Use comma-separated numbers without brackets.",f"{n}, {n+2}, {n+5}")
    for target,words in [(train,["brisa","colonia","nube","cobre","sendero","verde","flor","piedra"]),
                         (evaluation,["mariposa","montaña","silver","quiet"])]:
        for word in words:
            add(target,"exact",f"Escribe solo {word}. No añadas comillas, explicaciones ni signos.",word)
            add(target,"exact",f"Write only {word}. Add no quotes, explanations or punctuation.",word)
            for field in ["color","label"]:
                data=json.dumps({"name":"fixture","kind":"public","content":{field:word}},ensure_ascii=False)
                add(target,"extraction",f"Del JSON {data}, devuelve solo el valor de content.{field}, sin comillas.",word)
                add(target,"extraction",f"From JSON {data}, return only the value of content.{field}, without quotes.",word)
    for target,names in [(train,["Cedro","Cobre","Ámbar","Nube","Brisa","Lago","Monte","Río"]),
                         (evaluation,["Roble","Sombra","Perla","Valle"])]:
        for name in names:
            for yes in [True,False]:
                es_fact=f"{name} puede calcular actualizaciones." if yes else f"{name} transporta mensajes, pero no puede calcular actualizaciones."
                en_fact=f"{name} can compute updates." if yes else f"{name} transports messages, but cannot compute updates."
                add(target,"context",f"Dato: {es_fact} ¿Puede {name} calcular actualizaciones? Contesta solo sí o no, sin explicación.","sí" if yes else "no")
                add(target,"context",f"Fact: {en_fact} Can {name} compute updates? Answer only yes or no, with no explanation.","yes" if yes else "no")
                if yes:
                    add(target,"context",f"Dato: {name} aloja varios expertos. ¿Puede alojar más de un experto? Responde solo sí o no.","sí")
                    add(target,"context",f"Fact: {name} hosts several experts. Can it host more than one expert? Reply only yes or no.","yes")
    facts=[
        ("¿Qué es un nodo en una red?","Un nodo es un equipo o participante conectado que puede intercambiar información."),
        ("What is a node in a network?","A node is a connected computer or participant that can exchange information."),
        ("¿Para qué sirve una réplica del conocimiento?","Una réplica conserva una copia para que el conocimiento siga disponible si otro equipo se desconecta."),
        ("Why replicate knowledge?","A replica keeps a copy so knowledge remains available when another computer disconnects."),
        ("¿Qué significa evaluar una mejora antes de aceptarla?","Significa comparar la nueva versión con la anterior usando criterios fijados antes del entrenamiento."),
        ("What does evaluating before accepting an update mean?","It means comparing the new version with the previous one using criteria fixed before training."),
        ("¿Se pueden mezclar directamente gradientes de modelos con formas distintas?","No. Los modelos con formas distintas necesitan una ruta de transferencia compatible, como la destilación."),
        ("Can gradients from differently shaped models be combined directly?","No. Differently shaped models need a compatible transfer path, such as distillation."),
    ]
    for prompt,answer in facts:add(train,"explanation",prompt,answer)
    random.Random(20261014).shuffle(train)
    train_prompts={row["prompt"] for row in train}
    if len(train_prompts)!=len(train) or any(row["prompt"] in train_prompts for row in evaluation):
        raise ValueError("Public training and evaluation prompts overlap")
    retention=[
        "Computers exchange messages over networks. A network can contain many computers.",
        "Los equipos intercambian mensajes por redes. Una red puede contener muchos equipos.",
        "Plants use light, water and carbon dioxide during photosynthesis.",
        "Las plantas utilizan luz, agua y dióxido de carbono durante la fotosíntesis.",
        "Berlin is the capital of Germany. Rome is the capital of Italy.",
        "Berlín es la capital de Alemania. Roma es la capital de Italia.",
        "The moon orbits the Earth. The Earth orbits the Sun.",
        "La Luna orbita la Tierra. La Tierra orbita el Sol.",
        "A triangle has three sides. A square has four equal sides.",
        "Un triángulo tiene tres lados. Un cuadrado tiene cuatro lados iguales.",
        "An update should be evaluated before it replaces an earlier version.",
        "Una actualización se evalúa antes de sustituir la versión anterior.",
    ]
    result={"version":version,"scope":"public-synthetic-only","train":train,"evaluation":evaluation}
    if version==DATASET_VERSION:result["retention"]=retention
    return result


def corpus_digest(corpus):
    return hashlib.sha256(json.dumps(corpus,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
