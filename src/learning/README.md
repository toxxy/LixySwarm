# Aprendizaje público de expertos conversacionales

La red de referencia de bytes conserva su catálogo y sus versiones. Este módulo prepara expertos de lenguaje con bases públicas fijadas por revisión y SHA-256, sin importar checkpoints personales. Las operaciones de tensores corren en procesos separados; el nodo y la app importan solo configuración y transporte.

Se usa una actualización LoRA en las proyecciones de atención Q/V. Los pesos de la base quedan congelados y la actualización se guarda en safetensors con su hash de base, formas y configuración. La carga rechaza tensores incompatibles o no finitos antes de copiar alguno. Esta separación permite compartir actualizaciones pequeñas y conservar la base como referencia. La técnica se describe en el [paper original de LoRA](https://arxiv.org/abs/2106.09685).

## Evaluación y conservación

`public_conversation.py` genera 200 ejemplos sintéticos de instrucciones y 72 ejemplos con valores distintos para evaluación. La versión 2 añade doce textos públicos para penalizar diferencias respecto de la distribución de la base. No abre archivos de conversaciones, memorias, navegación ni datasets personales.

El primer experimento mejoró 7→11 respuestas correctas de doce, pero cambió una respuesta factual antes correcta. Se rechazó. La evaluación exige mejora de pérdida, al menos 11/12 en el criterio de instrucciones, todas las comprobaciones obligatorias y ninguna regresión de las respuestas antes correctas. El criterio inicial no basta para habilitar conversación general.

| Base | Adaptación | Criterio inicial | Evaluación ampliada | Alcance |
|---|---:|---:|---:|---|
| Qwen3-0.6B | 4.599.712 bytes | 7→12 / 12 | 13→23 / 28 | Conservó lo antes correcto, pero mantiene errores de contenido y extracción. |
| Qwen3-1.7B | 6.434.736 bytes | 8→12 / 12 | 16→26 / 28 | Las dos salidas restantes agregaron explicación; la revisión de esta muestra encontró correcto su contenido. |

Las 28 comprobaciones ampliadas usan prompts nuevos, contexto sintético y verificaciones por palabras o formato. Se guardan todas las salidas para revisión; estas métricas no prueban conversación general ni estabilidad. Algunos temas también están presentes en los textos públicos de conservación. Ambos entrenamientos fueron locales, sin reclamar otro ciclo de entrenamiento distribuido.

## Uso por Internet

`LanguageComputeService` registra `language.inference.v1` en el coordinador del nodo físico. El solicitante envía un identificador exacto de adaptación y mensajes explícitos de usuario/asistente, con límites de longitud y tokens. Nunca selecciona comandos, rutas, checkpoints o política. El hijo verifica base y adaptación, carga código de modelos únicamente desde la implementación local y no recibe clave de transporte, Matriarca ni almacén personal.

La reserva mínima de RAM considera el arranque: una petición del experto mayor superó 3 GiB y fue detenida. Se midió un máximo de 4.371.484.672 bytes de RSS; se requieren 5 GiB para ese experto y 4 GiB para el menor. La GPU y el número de procesos también deben caber en la política local. Los permisos se vuelven a comprobar durante el cálculo; retirar permisos cancela el proceso.

El piloto pidió tres respuestas desde la laptop al experto mayor alojado en la RTX 5090, bajo la identidad física existente, por Internet. Se verificaron recibos, modelo exacto y ruta cifrada. El solicitante no importó PyTorch ni Transformers. Son consultas a conocimiento aprendido localmente, no entrenamiento distribuido de lenguaje ni replicas nuevas. La configuración quedó preparada y deshabilitada después de restaurar los permisos originales.

Pendiente: familias con base y adaptaciones adquiridas por LSP, entrenamiento distribuido de estas adaptaciones, transferencia evaluada entre tamaños, selección de especialistas, chat desde la app y memoria persistente. La política personal sigue siendo responsabilidad del software; no se delega a las respuestas del modelo.
