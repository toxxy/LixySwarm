# Aprendizaje del organismo propio y expertos de investigación

La línea propia comienza en `swarm_foundation.py`: tres agentes de la arquitectura del enjambre, cinco pings Dolphin y feromonas por prefijo, inicializados con pesos aleatorios reproducibles. `swarm_foundation_job.py` ejecuta el primer preentrenamiento causal acotado y conserva pesos, optimizador AdamW y estado del muestreador. `swarm_foundation_audit.py` reproduce la evaluación en CPU y comprueba que el aprendizaje puede continuar desde el estado recibido. El protocolo, los límites y la evidencia están en [own_foundation_protocol_20261015.json](../../experiments/reports/own_foundation_protocol_20261015.json) y [own_foundation_internet_recovery_20261015.json](../../experiments/reports/own_foundation_internet_recovery_20261015.json).

Los profesores externos están permitidos como fuentes de destilación con procedencia y evaluación. Este primer ensayo usa solamente textos humanos públicos. No se presenta como aprendizaje distribuido, autonomía continua o chat general: esas integraciones siguen pendientes. La memoria persistente y los cambios de topología requieren un protocolo causal antes de habilitarse en este linaje; su recorrido actual los rechaza explícitamente.

La continuación distribuida usa `swarm_distributed.py` para fijar estado, fragmentos disjuntos, firmas, agregación y criterio de aceptación; `swarm_distributed_job.py` calcula gradientes y una candidata en hijos CPU. `OwnSwarmService` y `LocalOwnWork` integran estas operaciones en la identidad, IPC autenticado y presupuesto del nodo nativo. `experiments/own_swarm_round.py` inicia una ronda acotada mediante ese servicio. Es un controlador de operador, no un coordinador autónomo.

`swarm_state_audit.py` reconstruye una actualización desde sus gradientes firmados y el optimizador padre. `src/network/own_state.py` verifica transición, evaluación y auditoría antes de seleccionar pesos y optimizador juntos, conservando historia fuera de Dropbox. El [primer ensayo nativo conjunto](../../experiments/reports/own_swarm_native_round_20261015.json) calculó dos fragmentos distintos en las dos PCs, pasó el [criterio prefijado](../../experiments/reports/swarm_distributed_protocol_20261015.json), replicó por LSP los nuevos artefactos y verificó el mismo paso AdamW 1.025 en ambas. La laptop conservó la selección tras reiniciar su servicio. Esto acredita una actualización conjunta pequeña de predicción de bytes; siguen pendientes el ciclo continuo, el seguimiento automático de versiones, la entrada propia desde una instalación vacía y el chat del organismo.

El nuevo `OwnLearningCoordinator` integra la programación en el servicio nativo. Sus controles locales `own-learning-controls.json` comienzan deshabilitados: un coordinador conocido fija estado inicial, posición del corpus, fragmentos y hasta diez rondas. `own-learning/ledger.json` conserva contribuciones, candidata, réplicas y posiciones entre reinicios; cambiar el programa no reinicia ese registro. Tres rechazos consecutivos detienen el programa. Una pausa conserva el trabajo pendiente; un fallo de red reintenta la misma réplica sin contarlo como rechazo de calidad. Los rechazos de calidad consumen su rango de datos y mantienen los pesos anteriores.

`OwnStateReplication` sirve cabeza firmada mediante el protocolo de trabajo y recibe la transición del coordinador elegido localmente. Adquiere por LSP público cifrado pesos, optimizador, evaluación y ambos gradientes, verifica hashes y ejecuta `own-audit` en el motor CPU supervisado. Sólo entonces selecciona el estado. Si la laptop se quedó atrás, el coordinador recorre su historia local y envía las transiciones que le faltan. La próxima ronda espera las selecciones de ambas PCs. Este piloto sigue limitado a dos operadores conocidos y una coordinación elegida localmente; no implementa consenso abierto, elección de líder, metabolismo aprendido ni la preparación propia desde una app vacía. El [protocolo del primer programa continuo](../../experiments/reports/own_continuous_protocol_20261015.json) conserva esas limitaciones y la evaluación de desarrollo repetida; la prueba física de los ciclos nuevos se registra por separado.

El [programa físico A22](../../experiments/reports/own_continuous_native_20261015.json) completó diez ciclos nativos: siete mejoras replicadas y tres rechazos de calidad, conservando el paso AdamW 1.032 en ambas PCs y el cursor 2.816 en el coordinador. Pasó reinicio durante una réplica pendiente y pausa/reanudación sin reiniciar el progreso. La última versión aceptada mejoró la predicción de bytes del conjunto de desarrollo; el chat general y la alimentación constante con datos nuevos siguen pendientes. `python -m experiments.audit_own_cycles --node-directory "$HOME/.local/state/lixyswarm"` comprueba recibos físicos, rangos, gates, cadena, auditorías y estado final. `--laptop-report /ruta/al/informe.json` permite incorporar la comprobación compacta del segundo equipo, sin transferir sus pesos por administración.

## Expertos conversacionales con bases externas

El chat desplegado y los módulos siguientes pertenecen a los experimentos históricos con bases externas. Su funcionamiento no acredita por sí solo la visión del organismo con pesos propios.

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

El chat del piloto ya consulta desde la app y recupera historial cifrado. La política personal sigue siendo responsabilidad del software; no se delega a las respuestas del modelo.

## Paquetes de familia y réplica

El catálogo inicial `src/release/language_catalog.py` fija dos releases firmadas. Cada una vincula adaptación safetensors, descriptor, tokenizer y evaluación pública. El descriptor enumera todas las dependencias de la base por nombre cerrado, tamaño y SHA-256; conserva la licencia upstream. `publish_language_families.py` verifica los archivos públicos antes de importar y reproduce el genesis existente sin sustituirlo. No publica checkpoints personales ni selecciona archivos de memoria o conversación.

`LanguageFamilies` comparte la identidad y el coordinador del nodo físico. Sus operaciones `language.family.v1` y `language.artifact.*` conservan el protocolo cifrado, recibos y transferencia reanudable. El receptor verifica el genesis fijado y rechaza un tamaño diferente antes de solicitar chunks. Solo anuncia una familia completa después de verificar todos los objetos. La base mayor conserva sus dos shards originales; los gradientes de familias distintas no se mezclan.

Los paquetes públicos, motores y archivos parciales consumen la misma cuota de disco del participante. Reducirla por debajo del contenido conservado pausa el servicio público. El presupuesto de conexión y la pausa también se aplican a las transferencias de familia. La caché de respuestas de trabajo tiene un límite de 32 MiB, además del límite por cantidad de trabajos.

La primera réplica completa está en curso entre las PCs por Internet. La publicación inicial y una transferencia parcial no prueban disponibilidad independiente ni aprendizaje distribuido de lenguaje. Continúan pendientes la preparación del motor de lenguaje desde la app, la carga de una réplica adquirida, el entrenamiento distribuido de estas adaptaciones, la transferencia evaluada entre tamaños y la selección de especialistas.

## Versiones de adaptaciones y evaluación del piloto

`src/release/language_versions.py` conserva el genesis inmutable y descendientes con firma del publicador conocido, hash de pesos, dependencias upstream cerradas y evaluación por contenido. La cadena y sus archivos completos se verifican antes de cambiar la cabeza conservada. No se cambia el experto activo ni sus permisos al obtener una versión.

El gate fija por hash el corpus público y sus reglas: mejora de NLL, cuarenta paráfrasis factuales sin regresiones y cuarenta comprobaciones de instrucciones/contexto sin perder respuestas correctas. El segundo grupo exige al menos 37/40 y las instrucciones/contextos obligatorios del piloto. Los recibos deben corresponder a las dos PCs físicas conocidas, con versiones/datos compatibles, rangos distintos y trabajos sin duplicar. Una sola PC o repetir el mismo fragmento no acredita entrenamiento distribuido.

`experiments/publish_language_update.py` inspecciona una instantánea safetensors acotada sin importar PyTorch: nombres, formas, float32, rangos contiguos, metadatos del modelo público y valores finitos. La evaluación pasa antes de firmar o incorporar candidata/informe al espacio público. Requiere archivos fuera del checkout y la clave local del publicador existente; no genera ni distribuye autoridad nueva.

Los RPC `language.version-head.v1` y `language.version-history.v1` usan la identidad del nodo físico y su permiso de almacenamiento/conexión. El receptor obtiene toda la historia y las dependencias por LSP, verifica recibos, firmas y evaluaciones, y conserva el cambio de cabeza al reiniciar. Se rechazan forks y regresiones de secuencia. Las familias iniciales permanecen disponibles por el protocolo anterior.

Estas reglas corresponden al piloto de operadores conocidos y a evaluaciones elementales. No constituyen consenso abierto, pruebas contra publicadores maliciosos, calidad general del lenguaje ni un nuevo ciclo físico promovido. La coordinación de gradientes de lenguaje, su agregación real evaluada y la transferencia entre tamaños conservan sus gates pendientes.

`language.gradient.v2` vincula cada trabajo a la familia y a la cabeza verificada actual, además de adaptación, corpus y fragmento. El nodo materializa sus propios archivos verificados; el solicitante no elige rutas. Una actualización de cabeza durante el cálculo impide incorporar el resultado y reutilizar su caché. La familia mayor necesita una reserva mínima de 6 GiB de RAM y 6 GiB de VRAM por proceso; la menor requiere 4 GiB y 2 GiB respectivamente. Los controles antiguos se migran manteniendo sus permisos y su familia menor. El motor de lenguaje del código actual despacha inferencia y gradientes; los paquetes públicos anteriores mantienen su catálogo de inferencia y no se anuncian como motores de entrenamiento.

`language.candidate.v1` combina aportes con recibos físicos válidos, fragmentos disjuntos y hashes compatibles. Requiere habilitación local explícita (`candidates_enabled`) y el mismo permiso de recursos del trabajador. El hijo evalúa la adaptación anterior, aplica un paso SGD ponderado por tokens y evalúa la candidata con las mismas 72 muestras NLL, 40 paráfrasis factuales y 40 casos de instrucciones/contexto. El nodo comprueba de nuevo el resultado, conserva el informe y elimina los pesos si no cumplen todos los gates, incluido el de ambas PCs. No firma ni selecciona el nuevo experto. Los ensayos con una sola PC sirven para medir y rechazar candidatas, sin autorizar una promoción distribuida. Los directorios locales de candidatas también cuentan en la cuota de disco.

El chat selecciona la cabeza verificada de la familia mayor. Los clientes ligeros pueden adquirir únicamente la cadena de firmas y los informes de evaluación por LSP: comprueban procedencia, reglas de aceptación y hashes sin descargar pesos de modelos. Ese catálogo de conversación queda separado del registro de familias completas; no habilita cálculo local ni simula una réplica de pesos. Se rechazan forks, informes alterados y cadenas incompatibles con la versión local. Los anuncios distinguen los modelos disponibles para inferencia de los usados para entrenamiento.

`experiments/activate_language_version.py` selecciona una versión aceptada sobre un experto ya habilitado localmente. Repite la verificación de artefactos y evaluación, exige que el experto anterior siga siendo el esperado y conserva comando y permisos. Un cambio local durante la preparación cancela la selección. La inferencia mantiene los antecesores del experto seleccionado para clientes anteriores; adquirir una cabeza posterior no autoriza a servirla. La adaptación pequeña inicial continúa rechazada para la selección de chat.

`experiments/replicate_language_version.py` adquiere o repara versiones por LSP con uno o varios proveedores. Conserva la cabeza local si es más reciente, rechaza forks firmados y obtiene primero las evaluaciones de las transiciones. Los objetos faltantes o corruptos se descargan por hash con reanudación y cambio de proveedor. Una réplica parcial puede aportar objetos aunque no pueda anunciar una familia completa; si ningún historial remoto está disponible, solo se repara la cadena ya conocida localmente. El CLI comprueba consentimiento de almacenamiento/conexión, cuota y retirada de permisos, y usa una identidad de visitante separada de la del nodo físico. No cambia el experto activo ni sus recursos.

La preparación desde la app usa la misma adquisición de versiones y proveedores, incluidos los objetos complementarios de réplicas parciales. El RPC de cabeza sirve metadatos firmados aunque falte contenido local; el anuncio de familia completa conserva su comprobación independiente. Antes de seleccionar el experto, la app verifica cadena, objetos y evaluaciones, reserva almacenamiento para la versión elegida y el motor, y prueba el identificador exacto de esa adaptación bajo una reserva de recursos. Una cabeza o permisos que cambien durante la preparación impiden activarla. Las pruebas automatizadas usan fixtures y transporte cifrado local; la validación física del recorrido actualizado sigue pendiente.

La prueba física documentada en `experiments/reports/language_small_physical_repair_20261014.json` retiró temporalmente el adaptador Small de la laptop y lo recuperó desde la PC de escritorio por LSP público, verificando los mismos 4.599.712 bytes por hash y conservando permisos y experto activo. No demuestra aprendizaje nuevo, aprobación de Small para chat ni disponibilidad con el productor apagado. Las pruebas automatizadas de historias, evaluación y proveedores complementarios usan tensores sintéticos y no cuentan como réplicas físicas.

## Transferencia pública entre tamaños

`experiments/distill_public_language.py` explora enseñanza 1,7B→0,6B mediante texto, sin mezclar matrices o gradientes. Requiere una familia grande aprendida y aceptada ya adquirida por el alumno. El modo `collect` obtiene su historia y respuestas por LSP público desde un trabajador físico conocido; el modo `train` supervisa un hijo tensorial local sobre la familia pequeña original. No registra un experto ni modifica el coordinador o el catálogo activo.

```bash
python -m experiments.distill_public_language collect \
  --node-directory "$HOME/.local/state/lixyswarm" \
  --directory "$HOME/.local/state/public-lessons-new" --teacher NODE_ID
python -m experiments.distill_public_language train \
  --node-directory "$HOME/.local/state/lixyswarm" \
  --directory "$HOME/.local/state/public-student-new" \
  --lessons "$HOME/.local/state/public-lessons-new/lessons.json" \
  --engine-python /ruta/absoluta/al/python/local/con/CUDA
```

Los directorios de salida deben ser nuevos y permanecer fuera de Dropbox. El alumno comprueba etiquetas, firmas, ofertas originales, profesor y hashes; no entrena con respuestas incorrectas ni con entradas personales. Evalúa 32 entradas separadas y los controles factuales y de instrucciones existentes. `verify_student_report` vuelve a calcular el veredicto sin PyTorch. La política exige cuatro aciertos nuevos, NLL al menos 1 % menor y todos los controles de conservación; las candidatas rechazadas no conservan pesos. Las condiciones y limitaciones del [primer ensayo físico](../../experiments/reports/language_cross_size_transfer_20261014.json), incluido el fallo de impresión posterior al cálculo, se explican en [VISION_PROGRESS](../../VISION_PROGRESS.md).

## Medición del aporte de más PCs

`language_gradient_probe.py --shards 2 --count 8 --parallel-jobs 2` permite comparar dieciséis ejemplos en una PC con esos mismos fragmentos entre dos PCs. Mantiene la versión y permite usar las dos plazas GPU ya autorizadas del escritorio en la referencia de una sola PC. Cada proceso solicitante adquiere un bloqueo por identidad, incluso si conserva informes en otro directorio, para no interferir con un piloto que use la misma identidad de red.

`experiments/compare_language_scaling.py` verifica firmas, cadena, datos, rangos, tokens y correspondencia con los tiempos observados. Rechaza trabajos distintos o incompletos y muestra también una pérdida de rendimiento si añadir una PC resulta más lento. Mide el tiempo de cálculo y adquisición de gradientes después de preparar los modelos; excluye la descarga inicial y la evaluación de candidatas. Una medición aislada no acredita escalado general ni una mejora de modelo aceptada. Deben conservarse los permisos y registrarse otras cargas que puedan sesgar la comparación; una referencia obtenida durante la copia inicial se trata como calentamiento y se repite después de terminarla.

La ruta `OwnNetworkCoordinator` (A23, política v2) elimina la pareja fija: descubre cabezas firmadas compatibles y localmente admitidas, rota ventanas de hasta ocho aportes, mide capacidad por trabajos completados y reasigna fragmentos pendientes al perder participantes. Conserva recibos y candidata al reiniciar. Las réplicas alcanzan el linaje antes de aportar; la distribución posterior no espera a toda la colonia y continúa cuando termina el programa de datos. La política v1 y sus verificadores históricos mantienen su definición. Los controles `own-network-controls.json` y su progreso viven fuera del checkout y no reemplazan `own-learning/`. Esto aún requiere el motor y el estado propio preparados localmente, un coordinador seleccionado por el operador y el corpus público fijado; no demuestra preparación automática de una PC vacía, consenso abierto ni elección de líder.
