# Avance del goal de LixySwarm

El usuario activó [PLAN_VISION.md](PLAN_VISION.md) como goal el 2026-10-13.
Las etapas se completan por sus pruebas de aceptación; una implementación parcial conserva la etapa pendiente.

| Etapa | Estado | Evidencia y trabajo restante |
|---|---|---|
| 1. Red continua | En observación | Servicios desplegados en dos PCs y VPS, estado por identidad en el Explorer y recuperación tras reinicio comprobados. Faltan operación de 24 horas y recuperación con otro relay público independiente. |
| 2. App sencilla | En implementación | Paquetes y motor CPU verificados en tres sistemas; app instalada en ambas PCs. Chat habilitado con proveedor en el escritorio y cliente en la laptop, con historial cifrado. Faltan preparación y distribución automáticas del experto de lenguaje, activación de motores CUDA, arranque automático por plataforma y gate completo de instalación limpia. |
| 3. Aprendizaje distribuido | Ciclo de referencia integrado y verificado | Dos GPUs, coordinación desde la app y nodos persistentes, actualización candidata evaluada, publicación y recuperación de fragmentos tras reinicio. La referencia llegó a versión 6 y pérdida 5,38659. Faltan modelo conversacional, evaluación de calidad más amplia e instalación automática de motores. |
| 4. Conocimiento persistente | Piloto verificado parcialmente | Ambos trabajadores replicaron el modelo; un cliente vacío lo recuperó desde la otra PC con los procesos productores detenidos y reprodujo la evaluación. Faltan apagado físico, descubrimiento y reparación automáticos, memoria global y tres réplicas independientes. |
| 5. Capacidad heterogénea | Pendiente | Falta medir beneficio de dos PCs y transferencia entre agentes de tamaños distintos. |
| 6. Organismo funcional | Pendiente | Faltan integración persistente y comparaciones que demuestren utilidad de memoria, Dolphin y sectas. |
| 7. Autonomía | Pendiente | Faltan decisiones por resultados, GrowthGate y diez ciclos auditables. |
| 8. Red abierta | Pendiente | Faltan paquetes verificados, operadores independientes y condiciones de operación pública. |

## Decisiones del primer tramo

- **Identidad por máquina fuera de Dropbox.** El código puede sincronizarse, pero las claves, pares y estado de cada PC deben permanecer independientes. La alternativa de guardar identidad en el checkout confunde dos máquinas al sincronizarlas. Se comprueba con identidades distintas y conservación tras reinicio.
- **Telemetría pública tipada por LSP.** Cada nodo firma un esquema acotado de estado, recursos, contadores y rutas sin direcciones, conversaciones ni memoria. El collector conserva un registro por identidad y su propia observación de conexión; el Explorer lee esos registros. Se evita el archivo único y el token HTTP compartido entre participantes. Firma, caducidad, suplantación, repetición y persistencia tienen pruebas.
- **Recursos anunciados separados de capacidad comprobada.** La firma demuestra autoría. Los recursos y contadores reportados no se presentan como trabajo validado ni como mejora de aprendizaje.
- **Primer servicio de conectividad.** Las PCs anuncian entrenamiento e inferencia deshabilitados mientras se integra el ciclo real. No cargan modelos o memoria. Los trabajos reportados son cero en esta versión.
- **Linux como instalación de referencia.** El primer servicio usa systemd de usuario con arranque y recuperación. La app y los paquetes Windows/macOS conservan su etapa pendiente.

## Evidencia del despliegue inicial

Comprobación del 2026-10-13, finalizada el **2026-10-14 03:45:25 UTC**:

- La API pública y el navegador mostraron tres nodos activos: un seed/relay y dos participantes, con estado firmado y actualizado individualmente.
- Las PCs conservaron identidades distintas y anunciaron aproximadamente 32 y 8 GB de VRAM. Cada PC estableció una sesión directa con el seed público y una sesión entre participantes por relay; las rutas de salida usaron Ethernet y Wi-Fi, respectivamente.
- El reinicio limpio del servicio de la primera PC conservó su identidad, aumentó la secuencia persistente y recuperó sus dos sesiones.
- La terminación forzada del proceso de la segunda PC se reflejó como desconexión en la API. Systemd recuperó automáticamente el nodo, con la misma identidad y sus rutas públicas.
- Un ensayo separado de reinicio del seed recuperó las tres identidades y las dos rutas reportadas por relay en 10,7 segundos. Esto comprueba recuperación del mismo servicio; la sustitución por otro relay público sigue pendiente.
- Los clientes usaron aproximadamente 34 MiB de RAM cada uno, sin contexto de cálculo GPU. Publicaron cero trabajos; el entrenamiento permanece pendiente.
- Pasaron 91 pruebas de transporte, telemetría, trabajo, artefactos, versiones y planificación. La comprobación posterior sobre lectura de métricas caducadas también pasó, y el JavaScript del Explorer superó la comprobación de sintaxis.

Los servicios quedaron funcionando después de los ensayos. Desde esa comprobación se acumula evidencia para la ventana de 24 horas. El journal del seed registra cada 15 segundos conexiones, nodos con estado fresco, participantes y enlaces por relay; los journals de las PCs conservan estado y secuencias. Una interrupción requiere revisar la ventana antes de darla por cumplida.

## Uso del servicio de referencia

La app en desarrollo y su empaquetado se documentan en [desktop/README.md](desktop/README.md). La versión Linux se conectó al bootstrap público en 0,67 segundos durante el ensayo del ejecutable, confirmó límites, mantuvo la conexión al pausar y conservó identidad al reconectar. El transporte usó Ethernet hacia la dirección pública. El paquete ocupa aproximadamente 169 MiB antes de compresión e incluye el intérprete; los motores de cálculo permanecen excluidos. La persistencia cifrada se comprobó con el almacén protegido real de la PC de referencia.

La elección de Qt queda respaldada por el empaquetado y la prueba del ejecutable en las tres plataformas. Los presupuestos de CPU/GPU preparados en la interfaz todavía deben comprobarse con trabajadores de cálculo reales. No se marca la etapa como completada por los controles o los paquetes.

### Comprobación de la app del 2026-10-14

- [CI del código f6512bb](https://github.com/toxxy/LixySwarm/actions/runs/38517209677) completó los tres paquetes con éxito. En cada sistema se probó el ejecutable con un nodo separado y estado vacío: conexión cifrada, límites recibidos, pausa, cierre de sesiones y reconexión con identidad conservada.
- Se incluyeron las bibliotecas gráficas genéricas que faltaban en Ubuntu 22.04. La referencia nanoGPT ya tenía un gitlink; se documentó su origen existente en `.gitmodules` para permitir checkouts limpios.
- La comprobación estricta del cierre expuso una escritura bloqueada ocasionalmente por lectores en Windows. El reemplazo atómico ahora reintenta el bloqueo transitorio con espera acotada y conserva el registro completo anterior.
- Las dos PCs ejecutaron el paquete por Internet, con conexiones observadas en 0,67 y 0,76 segundos. Ambas confirmaron límites, mantuvieron conectividad al pausar y conservaron identidad al reconectar.
- La instalación por usuario creó el lanzador LixySwarm en el menú de aplicaciones de ambas PCs. El directorio de la app y el del estado son independientes.
- Los servicios de las PCs se actualizaron para que la interfaz pueda adjuntarse al nodo existente. Confirmaron configuración local con permiso GPU cero en el perfil inicial de conectividad. El despliegue realizó reinicios controlados durante la observación de red; la ventana de continuidad debe revisarse con esos eventos registrados.
- Pasaron 107 pruebas del proyecto. El chat y el cálculo continúan en preparación; la generación de claves y el cifrado personal están preparados para la integración de memoria.

El siguiente ensayo de referencia completó el cálculo, la evaluación y la adquisición por LSP. La conservación sin procesos productores y la integración en la app siguen sus propias condiciones de aceptación.

### Decisiones del ciclo de aprendizaje

- La pérdida principal corresponde a los logits combinados que entrega el enjambre. En entrenamiento se añade una pérdida auxiliar de agentes con peso 0,25 y una penalización diferenciable de similitud de feromonas con peso 0,01 y margen 0,7. La evaluación reporta la pérdida predictiva colectiva. Estos pesos son valores iniciales configurables; su efecto de calidad y especialización debe medirse.
- El entrenamiento productivo reparte rangos sin solapamiento y combina sus gradientes por cantidad de tokens predichos. La repetición del mismo rango mantiene su función de verificación en el agregador de mediana existente. No se relajan sus requisitos de quórum.
- El primer piloto productivo fija explícitamente dos identidades de trabajadores conocidos. La aceptación abierta de desconocidos conserva los gates de validación pendientes.
- Los cambios se aplican sobre un modelo candidato separado. Se conservan pesos y optimizador al promoverlo, con informe de evaluación y recibos en una release firmada. Una regresión conserva la versión activa. Una promoción interrumpida se recupera sin volver a aplicar gradientes.
- `experiments/internet_learning.py` usa un enjambre de referencia pequeño y reproducible, con tres agentes, Dolphin y feromonas. Su corpus sintético y su evaluación quedan separados; no carga memoria personal ni historial de especialización del checkout. El ensayo inicial prueba el recorrido y la pérdida de lenguaje; el modelo conversacional sigue pendiente.
- La adquisición de una versión avanzada conserva la activación desde genesis: un proveedor replica la cadena de manifiestos en páginas acotadas. El receptor verifica firmas del publicador, cada predecesor y el genesis fijado, además del recibo del proveedor. Se evita confiar en un proveedor para autorizar pesos o saltar la cadena. La descarga de checkpoints precede a cada activación; compactar esa historia requiere un protocolo de instantáneas evaluado posteriormente.

### Aprendizaje por Internet del 2026-10-14

El [informe de referencia](experiments/reports/internet_learning_20261014.json) conserva los recibos y manifiestos públicos, sin direcciones de operadores. Los pesos y el estado del piloto están fuera de Dropbox; sus transferencias se hicieron por LSP.

- Dos trabajadores físicos: RTX 5090 y RTX 2000 Ada. Los procesos de ensayo usan identidades distintas de los servicios de conectividad; no representan PCs adicionales.
- Referencia de 245.676 parámetros, vocabulario de 256 bytes, tres agentes y cuatro rondas. Cada GPU calculó cuatro fragmentos distintos de 65 bytes, equivalentes a 64 predicciones por fragmento: 512 tokens predichos entre ambas.
- Pérdidas de evaluación sucesivas: 5,5608599235 → 5,5377551968 → 5,5123241738 → 5,4844239352 → 5,4548538764. Todas las rondas superaron la mejora mínima prefijada de 0,001. La reducción relativa de pérdida fue 1,906%; no es un porcentaje de comprensión o de calidad conversacional.
- La release final es `59f68cdc1e6985fd539c1299e42bd200a4a24c42959d368fc3748780bcea98a3`; los pesos y el optimizador tienen SHA-256 `f2e3556f06a392291157b75c6adc44378cc15e1b10ddb9f633a2da19d30c66c2`.
- Las GPUs adquirieron la misma versión y respondieron a la consulta de logits del siguiente byte. Su diferencia máxima fue 2,98 × 10⁻⁸. Esto verifica uso equivalente del modelo; el modelo pequeño todavía no entrega un chat útil.
- Las sesiones entre trabajadores y coordinador atravesaron el relay público con cifrado de extremo a extremo. El coordinador corrió en una PC; el VPS transportó tráfico y no entrenó ni decidió las promociones.

El corpus de evaluación comparte los hechos sintéticos de entrenamiento, con contexto distinto. Este gate comprueba una mejora reproducible de predicción y el recorrido distribuido; la generalización semántica y el aumento de capacidad al sumar nodos requieren otras evaluaciones.

### Conservación y adquisición desde cero

- Se cerraron limpiamente el coordinador y el trabajador de la PC productora. La otra PC reinició su proceso y restauró la release final, la identidad y los cuatro trabajos completados desde su estado persistente.
- Un cliente nuevo, con cero bytes de artefactos, obtuvo desde esa PC cinco manifiestos firmados y 9.243.701 bytes de pesos, optimizador y evaluaciones por LSP. Verificó el genesis fijado, las firmas, cada predecesor y los hashes antes de activarlos.
- Ese cliente cargó los pesos en CPU y reprodujo la evaluación: 5,4548539453 frente a 5,4548538764 en GPU. La diferencia fue inferior a 10⁻⁷.
- El [informe de recuperación](experiments/reports/internet_persistence_20261014.json) conserva la evidencia sin direcciones de operadores. Las rutas hacia el relay público usaron Ethernet y Wi-Fi; la VPN se utilizó solamente para administrar procesos.
- La prueba conserva acceso a los servicios de conectividad de las dos PCs, que no sirven estos checkpoints. Comprueba independencia de los procesos productores y almacenamiento vacío; apagar físicamente la PC productora sigue pendiente y no se da por cumplido con este ensayo.
- Pasaron 110 pruebas de aprendizaje, versiones, artefactos, trabajo, transporte, telemetría, interfaz y límites. Los ensayos temporales se mantienen separados del cliente de uso diario.

### Recuperación de conexiones sin respuesta

El despliegue posterior al piloto expuso una conexión TCP de la primera PC que permanecía abierta con datos sin confirmar después de reiniciar el seed. La interfaz local conservaba pares antiguos mientras el collector ya había caducado su estado. Esta observación invalida dar por terminada la continuidad con los ensayos anteriores.

Se añadieron pings cifrados cada 15 segundos, cierre tras 60 segundos sin recibir una trama válida y espera máxima de cinco segundos para enviar una trama. El cierre elimina la sesión y sus túneles, notifica trabajos pendientes y despierta el descubrimiento para volver a conectar. La alternativa de esperar el timeout TCP del sistema conserva sesiones sin respuesta durante demasiado tiempo. Una prueba con un proxy TCP que descarta todos los paquetes, manteniendo sockets abiertos, comprobó caducidad y reconexión cifrada sin reiniciar el nodo; una conexión inactiva sana sobrevivió varias ventanas.

El [ensayo público de interrupción](experiments/reports/internet_liveness_20261014.json) descartó durante 80 segundos únicamente el tráfico TCP LSP de una PC. El nodo pasó de conectado a reconectando y volvió al seed con el mismo proceso e identidad, dentro de los seis segundos observados tras retirar las reglas. Recuperó también el enlace cifrado con la otra PC por relay. Las reglas temporales se retiraron automáticamente y su ausencia se verificó; al finalizar quedaron tres servicios activos y frescos. Este ensayo sigue usando un único relay, por lo que no cumple el gate de sustitución por otro operador.

Pasaron 111 pruebas, incluyendo pérdida silenciosa de paquetes. El [CI 59a26a3](https://github.com/toxxy/LixySwarm/actions/runs/38523688292) reconstruyó y verificó los tres paquetes. Las apps Linux actualizadas se probaron por Internet en ambas PCs, con conexiones de 0,72 y 0,76 segundos y controles, pausa e identidad conservados. Los lanzadores del menú apuntan a los nuevos paquetes. La observación de continuidad debe empezar después de esta interrupción controlada; todavía no han transcurrido 24 horas.

Próximo tramo del goal: integrar el almacenamiento público y los trabajadores como procesos separados del servicio de cada PC, con una sola identidad por equipo; incorporar adquisición de modelos y uso desde la app. La referencia de bytes conservará su función de evaluación del protocolo, y el modelo conversacional tendrá su propia evaluación de calidad.

### Integración de conocimiento público en el servicio

- El servicio ligero incorpora artefactos, historia firmada y adquisición del catálogo usando la identidad persistente del equipo. Los pesos se conservan y sirven sin importar PyTorch. La inferencia y los trabajadores de cálculo mantienen su integración pendiente.
- El catálogo inicial fija el genesis y el publicador del piloto conocido; un proveedor de archivos no obtiene autoridad para aceptar modelos. La alternativa de descargar el modelo más anunciado sin política de firmas permite suplantación. La política abierta entre operadores sigue pendiente.
- La app ofrece **Conservar conocimiento público**, con consentimiento separado para almacenamiento, progreso y pausa. El almacén `public/` sirve solamente archivos incorporados por el recorrido de releases; nunca abre el banco personal ni copia todo el directorio de un ensayo.
- Las transferencias comparten un presupuesto de chunks codificados, interrumpible y limitado a la cuota elegida y a 5 Mbps durante esta integración. Las cuotas se aplican antes de nuevas adquisiciones; una reducción conserva los archivos existentes y suspende su servicio. Los nodos sin GPU pueden colaborar almacenando versiones.
- Las pruebas cubren adquisición desde cero, continuidad de identidad del proveedor, restauración de versiones, pausa conservando pesos, cancelación, cuotas y exclusión de contenido personal de una migración. La comprobación en PCs reales y ejecutables empaquetados se registra al finalizar el despliegue.

### Verificación de almacenamiento y app del 2026-10-14

- Los dos servicios de las PCs migraron solamente los artefactos de las cinco releases públicas del piloto. Conservan sus identidades originales y sirven 9.243.701 bytes por LSP. No se migraron claves del piloto, datos privados ni todo su directorio de estado.
- Después de reiniciar ambos servicios, sus diagnósticos muestran la misma release final y la cadena conservada. Ocupan aproximadamente 35 MiB de RAM por proceso y no tienen bibliotecas de PyTorch cargadas. El permiso de cálculo GPU continúa en cero.
- El [CI 1d01777](https://github.com/toxxy/LixySwarm/actions/runs/38526456146) verificó paquetes Linux, Windows y macOS, con pruebas adicionales de adquisición, firma, controles y almacenamiento público sin motores de cálculo.
- En ambas PCs, el ejecutable instalado se probó con directorio vacío y el botón de conservación. Adquirió por Internet la cadena y los pesos, verificó el hash de la release, conservó los archivos al pausar y mantuvo identidad y versión al reconectar. Las conexiones iniciales se observaron en 1,62 y 0,82 segundos.
- El [informe de los ejecutables](experiments/reports/desktop_public_knowledge_20261014.json) guarda esos resultados. Ambos menús de aplicaciones apuntan a los nuevos paquetes. La inspección visual comprobó que estado, conservación, pausa y conexión caben en la ventana.
- Pasaron 116 pruebas del proyecto; la comprobación posterior de disponibilidad anunciada y estado de la interfaz pasó 47 pruebas pertinentes. El VPS se actualizó por Git y conserva exclusivamente transporte, descubrimiento, API y Explorer.

Los participantes con esta app pueden obtener y conservar la referencia sin terminal ni GPU. Usarla para cálculo y ofrecer conversación requieren todavía los motores separados y su evaluación. El catálogo inicial permanece limitado al piloto conocido; no se da por terminado el conocimiento global, la diversidad de modelos ni la aceptación abierta de aportes.

### Procesos supervisados de cálculo

- El nodo físico conserva la identidad, las conexiones, los artefactos y los recibos. Un proceso hijo sin identidad de red carga únicamente el checkpoint público activo y devuelve inferencia de referencia o un gradiente. El servicio ligero permanece sin PyTorch.
- El supervisor termina el proceso al cancelar, vencer el plazo, retirar permisos, cambiar el presupuesto, exceder RAM o producir salida fuera de límites. Los resultados temporales se incorporan por hash y sus directorios se retiran al finalizar. Los contadores de cálculo completado se conservan entre reinicios y son distintos de trabajo aceptado por aprendizaje.
- El primer motor acepta el corpus público fijado y rangos de hasta 65 tokens; entrenamiento solo desde los coordinadores conocidos del piloto. La inferencia de bytes no se anuncia como conversación útil. Otros modelos y el protocolo abierto requieren sus validaciones posteriores.
- El motor se configura localmente; un peer no puede enviar un comando o implementación ejecutable. Las dos PCs permiten probarlo con los entornos de cálculo existentes. El instalador automático y los paquetes de motores CPU/GPU continúan pendientes; el paquete ligero no incorpora ese entorno de desarrollo.
- Esta separación permite detener un cálculo sin perder la identidad o la versión aceptada. Es supervisión de procesos y permisos; el aislamiento completo del sistema de archivos y los límites generales de utilización GPU conservan su etapa de endurecimiento pendiente.
- Las capacidades cambian con los permisos y los motores disponibles. Un anuncio acotado en los pings cifrados refresca capacidades, modelos y presupuestos sin reiniciar sesiones; conserva compatibilidad con pares anteriores. La evidencia de trabajo verificada localmente se mantiene separada y no puede sobrescribirse con ese anuncio.
- La primera prueba CUDA detuvo un gradiente que superó la reserva declarada de 1,5 GiB. Solicitar 2 GiB, dentro del perfil equilibrado de prueba, completó el trabajo. El requisito incluye el runtime de CUDA, además del pequeño modelo; no se eliminó la comprobación de RAM para lograr el ensayo.
- Un reintento de un trabajo completado vuelve a comprobar consentimiento y permisos de la operación. Las respuestas de chunks guardadas también consumen presupuesto de transferencia. Se conserva deduplicación sin usar la caché para continuar un aporte retirado.

### Verificación de cálculo en los nodos persistentes

El [ensayo por Internet](experiments/reports/persistent_compute_20261014.json) ejecutó inferencia y cálculo real de gradientes en las GPUs de ambas PCs, bajo `c45094…` y `ea914d…`, sus identidades persistentes. El solicitante temporal desempeñó el papel de prueba y no se contó como otro trabajador físico.

- Cada trabajador procesó un fragmento diferente del corpus público. Se verificaron recibos, hashes, esquema NPZ, parámetros, formas y valores finitos antes de aceptar sus resultados de cálculo. Los gradientes quedaron como artefactos; este ensayo no promovió una nueva versión.
- Los tiempos observados en el último ensayo fueron 1,277/1,419 segundos para inferencia/gradiente en la RTX 5090 y 1,881/1,982 en la RTX 2000 Ada. Estas medidas incluyen arranque del motor y transporte; no demuestran todavía escalado del aprendizaje.
- Los logits de ambas GPUs difirieron como máximo en 2,98 × 10⁻⁸. La referencia conserva su evaluación de predicción de bytes y no se presenta como calidad conversacional.
- Durante otra petición, pausar el aporte terminó un motor CUDA confirmado vivo. El proceso del nodo, las conexiones, la identidad y los pesos se conservaron. Los límites iniciales de las PCs se restauraron al terminar: conectividad y GPU cero. Los motores locales quedaron preparados y el solicitante de prueba fue retirado de sus coordinadores autorizados.
- El servicio permaneció sin PyTorch. Tras servir transferencias y trabajos, la primera PC usó aproximadamente 80 MiB de RAM; la caché de resultados está acotada. Los contadores conservan solamente cálculos completados y la API sigue distinguiendo estos reportes de aportes validados o promovidos.

Pendiente para el siguiente recorrido completo: coordinación desde el nodo persistente, trabajos recuperables, evaluación y promoción que actualicen el catálogo; instalación automática de motores y modelo conversacional. El cálculo de referencia preparado en las dos PCs utiliza sus entornos existentes fuera del paquete ligero, y no da por cumplida la instalación limpia de motores en una PC nueva.

La comprobación posterior pasó 122 pruebas. El [CI dbfc2a0](https://github.com/toxxy/LixySwarm/actions/runs/38573547066) verificó los paquetes en Linux, Windows y macOS con los controles de procesos y permisos de caché. Los dos ejecutables Linux volvieron a adquirir el modelo desde directorios vacíos por Internet, conservando pesos e identidad al pausar y reconectar; el proceso ligero siguió sin cargar PyTorch. Sus lanzadores se actualizaron en ambas PCs. El VPS adquirió el código por Git y recuperó sus servicios de transporte y API.

### Coordinación y continuidad del aprendizaje

- El coordinador forma parte del servicio físico. Un fragmento local obtiene el mismo tipo de recibo y usa la misma identidad que su trabajo remoto; no se crea otra hormiga para coordinar. Los fragmentos se reparten entre las dos PCs conocidas.
- Un ledger registra base, posiciones, recibos, gradientes obtenidos, evaluación y promoción pendiente. Reiniciar conserva las contribuciones verificadas; cambiar la base descarta trabajo anterior incompatible. Una promoción pendiente se recupera sin aplicar de nuevo la actualización. La primera posición del coordinador se toma del informe aceptado cuando ya existe entrenamiento previo.
- La evaluación corre en un proceso supervisado sin clave de publicación. Devuelve una candidata sin firmar; el servicio verifica el criterio prefijado y firma con la clave local del publicador autorizado. Esa autoridad de la familia se mantiene separada de las identidades de transporte y cálculo. El piloto conserva una sola autoridad conocida; el consenso entre operadores sigue pendiente.
- El catálogo puede seguir descendientes firmados del genesis fijado. Los peers anuncian su versión actual al conectar y verifican la cadena antes de activar una actualización. Se conserva activación monótona y se rechazan publicadores desconocidos. Las actualizaciones no alteran los permisos de almacenamiento ni cálculo.
- La app incorpora el control de una ronda acotada, detención y estado del aprendizaje. La ventana inicial admite hasta diez rondas. La programación autónoma por utilidad y GrowthGate conservan sus etapas pendientes.

### Ensayo nativo y recuperación del 2026-10-14

El [informe de aprendizaje nativo](experiments/reports/native_learning_20261014.json) conserva versiones, evaluaciones y recibos de los trabajadores físicos.

- Una ronda iniciada desde el control de la app repartió rangos 260 y 7460. El coordinador persistente recibió los gradientes local y remoto, evaluó y firmó la versión 5. La pérdida fija pasó de 5,4548539453 a 5,4220078572. La otra PC adquirió y activó esa versión por LSP.
- En la siguiente ronda se detuvo el servicio de la segunda PC. La primera terminó y guardó su contribución. Se reinició el coordinador, se comprobó que conservaba el mismo recibo y se reanudó la segunda PC. La ronda recuperada utilizó ese recibo sin repetir el fragmento completado y promovió la versión 6: pérdida 5,3865938600.
- La versión 6 es `977603e7a2c2070d6cd1c705e53f763d7e31057d5980f36cc02aefb2bb7ce25c`; sus pesos y optimizador tienen hash `23e44d2c2f0834f798c4309b7f0360cae0b9d4769bb84a452498e95d322524a9`. Ambas PCs la conservaron. El cursor quedó en 390 y las dos rondas terminadas se mantuvieron en el ledger.
- El ensayo se realizó en los nodos físicos existentes, con las rutas públicas cifradas. La clave de publicación permaneció en la PC coordinadora; el evaluador no la recibió y el VPS no coordinó ni entrenó.
- Se restauraron los perfiles originales de conectividad y GPU cero y se detuvo la programación de rondas. La configuración del coordinador y los motores permanece preparada para la elección explícita de recursos y aprendizaje en la app.
- Pasaron 127 pruebas del proyecto. Un fallo nativo al cerrar el ejecutable Linux se reprodujo después de escribir el informe funcional. Se añadió diagnóstico y se corrigió el ciclo de vida de temporizadores y widgets Qt antes de terminar la aplicación. Los ejecutables corregidos volvieron a adquirir la versión 6 por Internet y terminaron con código cero en ambas PCs.

La evaluación sigue siendo predicción de bytes sobre hechos sintéticos conocidos. Estas promociones comprueban continuidad y aprendizaje distribuido; la calidad conversacional, el beneficio de escalar nodos y las comparaciones de memoria y especialización mantienen sus propios gates.

El [CI f806197](https://github.com/toxxy/LixySwarm/actions/runs/38579619373) completó los tres paquetes después de la corrección del cierre. Las pruebas de ambos ejecutables por Internet adquirieron la versión 6 y terminaron limpiamente. Una observación adicional detectó demora al recorrer direcciones antiguas de ensayos: el catálogo inicial añade dos identidades de proveedores operados como pistas de conexión directa por identidad, antes de continuar la búsqueda general. Estas pistas no otorgan autoridad sobre modelos ni sustituyen la validación de firmas. El reporte del ejecutable registra ahora el tiempo de adquisición además del tiempo de conexión.

La verificación final pasó 128 pruebas y el [CI b4e8bfd](https://github.com/toxxy/LixySwarm/actions/runs/38580594941) completó los tres paquetes. Los ejecutables Linux se conectaron en 0,72 y 0,76 segundos, y adquirieron 13.092.926 bytes de la cadena de la versión 6 en 58,58 y 66,34 segundos durante dos descargas concurrentes con presupuesto. La adquisición permaneció independiente de la conexión inicial. Conservación, pausa, reconexión y cierre pasaron; ambos lanzadores se actualizaron. Los servicios mantienen la versión 6, dos rondas terminadas en la primera PC y GPU cero al finalizar.

### Base conversacional y motor CPU empaquetado

- La base bilingüe propia `foundation_v1` genera texto, pero una comprobación inicial falló instrucciones y preguntas sencillas. Los checkpoints del fine-tuning personal no se publican ni se usan como semilla global.
- Se fijaron doce comprobaciones públicas de instrucciones, cálculo, traducción, extracción y contexto antes de evaluar dos semillas Qwen3. Las revisiones, hashes, prompts, respuestas y criterios permanecen en los informes: [0,6B greedy](experiments/reports/conversation_seed_06b_greedy_20261014.json), [0,6B muestreo](experiments/reports/conversation_seed_06b_sampling_20261014.json), [1,7B greedy](experiments/reports/conversation_seed_17b_greedy_20261014.json) y [1,7B muestreo](experiments/reports/conversation_seed_17b_sampling_20261014.json).
- La semilla de 0,6B obtuvo 7/12 y 6/12, incluyendo errores al usar el contexto suministrado. La de 1,7B obtuvo 8/12 con ambos modos: respondió correctamente al contenido, según revisión de estas doce salidas, pero añadió explicaciones, comillas o corchetes donde se pedía un formato exacto. El gate fijado exige 11/12 y las comprobaciones obligatorias; ninguna semilla lo superó. No se relajó el criterio para declarar calidad ni se habilitó el chat. Es una prueba elemental, no una evaluación general del lenguaje.
- Las semillas son pesos públicos Apache-2.0, cargados localmente sin código remoto ni una API de inferencia externa. Su consideración como punto de partida no demuestra aprendizaje del enjambre ni sustituye los experimentos de cooperación, memoria y transferencia entre tamaños. El siguiente trabajo debe corregir y medir estas limitaciones con datos públicos y evaluaciones separadas del entrenamiento.
- El [motor CPU separado](engines/README.md) incluye intérprete, PyTorch CPU y operaciones del modelo de referencia. El [ensayo del ejecutable](experiments/reports/cpu_engine_20261014.json) pasó inferencia, gradientes y evaluación de una candidata desde un directorio temporal, sin el Python ni las rutas de desarrollo en su ejecución. La candidata redujo la pérdida de 5,5608599235 a 5,5377550138 y regresó sin firma; la versión activa de la prueba permaneció intacta. Sus tres firmantes son fixtures locales de la prueba, no otros equipos físicos ni evidencia de aprendizaje por Internet. La verificación de instalación comprueba un digest local confiable y todos los archivos antes de seleccionar el ejecutable; conserva coordinadores autorizados y permisos de recursos. Cargar la versión pública 6 desde el motor instalado produjo los mismos logits CPU dentro de una reserva de 2 GiB.
- Un paquete CPU no anuncia CUDA, aunque la máquina tenga GPU. El coordinador elige CPU o CUDA según el motor local y los recursos anunciados por el trabajador. La app sigue siendo ligera. La descarga automática de motores, la distribución pública y la instalación física en equipos nuevos siguen pendientes; este ejecutable local no da por cumplida toda la etapa 2.

Pasaron 138 pruebas del proyecto y 18 pruebas relevantes en la segunda PC con el código sincronizado por Dropbox. El [CI de la app 3b60009](https://github.com/toxxy/LixySwarm/actions/runs/38585333232) y el [CI inicial del motor CPU](https://github.com/toxxy/LixySwarm/actions/runs/38585333266) completaron Linux, Windows y macOS. El [CI completo del motor adea2f8](https://github.com/toxxy/LixySwarm/actions/runs/38586287608) verificó además las candidatas sin firma ni promoción en las tres plataformas. El mismo ejecutable completo pasó en un contenedor Ubuntu sin Python instalado, sin acceso a red, con raíz de solo lectura y límites de dos CPU y 2 GiB de RAM. Este ensayo prueba el paquete CPU; no sustituye la instalación limpia de toda la app. El VPS adquirió el código por Git y conservó activos relay y API, sin instalar este motor ni alojar pesos.

### Preparación automática desde la app

- Los [motores CPU](https://github.com/toxxy/LixySwarm/releases/tag/cpu-engine-0.4.0a1) tienen descargas públicas sin cuenta. El catálogo dentro de la app fija URL, plataforma, arquitectura, tamaño y hashes del ZIP y manifiesto. Las operaciones siguen siendo las del modelo de bytes de referencia; el chat permanece pendiente.
- Guardar recursos de CPU con al menos 2 GiB de RAM inicia la preparación en segundo plano. Una entrada nueva habilita el conocimiento público para ese aporte; una retirada anterior del permiso de compartir se conserva. El motor CUDA local previo no se sustituye. Descargar el motor no amplía CPU, GPU, RAM ni coordinadores autorizados.
- La preparación muestra progreso, conserva el archivo parcial al pausar o desconectar, verifica el archivo completo y cada miembro antes de seleccionar el ejecutable y rechaza rutas que escapen, corrupción, archivos adicionales y tamaños incorrectos. Reserva la mitad del presupuesto de conexión; el servicio de conocimiento utiliza la otra mitad mientras la descarga está activa. Un marcador caducado libera esa reserva.
- El [recorrido del ejecutable por Internet](experiments/reports/desktop_engine_preparation_20261014.json) conectó en 0,72 segundos, adquirió la versión pública 6 por LSP y preparó el motor por HTTPS. Sus pruebas independientes de inferencia, gradientes y candidata pasaron dentro del ejecutable del motor. App y nodo siguieron sin PyTorch. Pasaron 147 pruebas del proyecto.
- El [CI d978ef1 de la app](https://github.com/toxxy/LixySwarm/actions/runs/38591726389) comprobó descarga pública, preparación y ejecución del motor desde los paquetes de Linux, Windows y macOS. El [CI del motor](https://github.com/toxxy/LixySwarm/actions/runs/38591726156) también pasó en las tres plataformas. Las pruebas físicas, la distribución CUDA y el modelo conversacional conservan sus gates. Los archivos públicos para instalar toda la app se preparan como archivos nativos que conservan los symlinks de los frameworks.

Desde el checkout y su entorno Python con `cryptography`:

```bash
.venv/bin/python client_daemon.py --install-user-service
systemctl --user status lixyswarm-node.service
journalctl --user -u lixyswarm-node.service
```

El estado se guarda en `$XDG_STATE_HOME/lixyswarm` o `~/.local/state/lixyswarm`.
Cada PC conserva su propia identidad. El servicio inicia con la sesión de usuario; `Linger=yes` permite que arranque sin sesión interactiva. La conexión usa los seeds públicos configurados y no necesita VPN.

El seed y la API comparten `LIXYSWARM_STATUS_DIRECTORY`, en un directorio local persistente legible por la API y escribible por el servicio seed. El collector firma su propio estado, verifica el de sus pares y expira registros después de 60 segundos. Al reiniciarse deja de contar sesiones anteriores hasta recibir estado nuevo.

`swarm_publisher.py` y su endpoint HTTP conservan compatibilidad con métricas de entrenamiento anteriores. La frescura de entrenamiento y de red se exponen por separado; datos de entrenamiento caducados no se presentan como actividad actual.

El recorrido ampliado verificó trabajo de CPU servido por la app en ambas PCs: [desktop](experiments/reports/desktop_engine_worker_desktop_20261014.json) y [laptop](experiments/reports/desktop_engine_worker_laptop_20261014.json). Conectaron en 0,72/0,77 segundos, adquirieron el modelo en 42,52/38,80 segundos y prepararon el motor en 53,04/59,36 segundos. Un solicitante de prueba por Internet recibió logits de la versión 6 y un recibo verificado del nodo de la app. Estas son instancias de prueba en las dos máquinas existentes; no se contabilizan como otros equipos físicos. Los motores auxiliares verificaron candidatas de fixtures, sin reclamar otro entrenamiento distribuido ni calidad conversacional.

Una prueba concurrente de la laptop excedió 180 segundos al adquirir el modelo; la repetición independiente completó la adquisición en 38,80 segundos. Ese incidente se conserva como observación de disponibilidad; no se infiere estabilidad prolongada del último éxito. La prueba limpia de la app en Ubuntu sin Python detectó dos bibliotecas gráficas ausentes; el paquete ahora incluye GL y XCB y pasó conexión, controles y cierre en el contenedor sin red externa. Las políticas productivas se conservaron: la laptop actualmente tiene el perfil equilibrado autorizado (25 % CPU/GPU); no se reemplazó por el perfil de conectividad observado anteriormente.

La [vista previa 0.4.0a2](https://github.com/toxxy/LixySwarm/releases/tag/v0.4.0a2) publica archivos nativos de Linux x86_64, Windows x86_64 y macOS arm64, con hashes y descarga sin cuenta. El [CI final 4eea22a](https://github.com/toxxy/LixySwarm/actions/runs/38597441890) verificó los tres paquetes con preparación pública del motor. Los archivos conservan los symlinks originales de frameworks. Los servicios y accesos de ambas PCs usan ahora los intérpretes de las apps instaladas, mantienen las mismas identidades, versión 6 y contadores 7/6; las unidades anteriores tienen respaldo local. No dependen del Python de desarrollo para la red. Los motores CUDA heredados se conservan y su reemplazo por paquetes públicos sigue pendiente.

### Expertos de lenguaje y consulta remota

La [implementación de adaptaciones públicas](src/learning/README.md) añade LoRA con base congelada, artefactos safetensors y corpus público sintético versionado. El [primer intento](experiments/reports/conversation_adapter_v1_20261014.json) alcanzó 11/12 pero perdió una respuesta factual previa; quedó rechazado. Con penalización de conservación, la [base menor](experiments/reports/conversation_adapter_v2_20261014.json) y la [mayor](experiments/reports/conversation_adapter_17b_v2_20261014.json) alcanzaron 12/12 sin regresiones en ese criterio.

La evaluación ampliada pasó 13→23 / 28 en la [base menor](experiments/reports/conversation_dialogue_06b_v2_20261014.json) y 16→26 / 28 en la [mayor](experiments/reports/conversation_dialogue_17b_v2_20261014.json). Todos los textos y resultados se conservan; las métricas por palabra o formato son limitadas y no prueban conversación general. Estos son entrenamientos locales; las actualizaciones pesan 4,6/6,4 MB y no sustituyen la base ni mezclan gradientes de modelos incompatibles.

El [piloto de lenguaje por Internet](experiments/reports/language_internet_20261014.json) consultó al experto mayor en la PC principal desde la laptop. Tres respuestas tardaron 4,97/5,07/4,87 segundos y sus recibos corresponden a la identidad física persistente `c45094…`; el cliente de prueba no cuenta como otro trabajador físico ni carga el modelo. La ruta fue cifrada y mediante el relay público. El VPS no recibió claves, modelos ni tareas de cálculo.

El [ensayo de recursos](experiments/reports/conversation_resources_20261014.json) detuvo un arranque que excedió la reserva de 3 GiB. La medición posterior del sistema registró 4.371.484.672 bytes de RSS y 4.064.542.720 bytes de asignación GPU. El experto mayor requiere una reserva de 5 GiB y GPU compatible; el servicio conserva cancelación y permisos durante el cálculo. Se restauraron los recursos originales de la PC principal y se dejó el experto preparado pero deshabilitado. No se alteró el perfil equilibrado de la laptop.

Pasaron 158 pruebas. El chat público de la app aún no se habilita: faltan el recorrido de usuario y memoria local, la publicación de familias con base y adaptación replicadas por LSP y el aprendizaje distribuido de lenguaje. La transferencia entre tamaños y las comparaciones de cooperación conservan sus gates.

El [CI del nodo/app 91418de](https://github.com/toxxy/LixySwarm/actions/runs/38607062855) y el [CI del motor CPU](https://github.com/toxxy/LixySwarm/actions/runs/38607063045) completaron Linux, Windows y macOS. Verificaron los controles ligeros, el empaquetado y las pruebas de adaptaciones en el entorno de tensores correspondiente. No ejecutaron inferencia del experto mayor en las tres plataformas; ese ensayo real pertenece a las dos PCs Linux. El VPS adquirió el código por Git y mantuvo activos relay y API, sin instalar bases de lenguaje ni adaptaciones.

### Chat de la app y memoria personal

El [ensayo del ejecutable 0.4.0a4](experiments/reports/chat_app_internet_20261014.json) abrió una app con estado vacío en la laptop, conectó al transporte público en 0,77 segundos y obtuvo respuestas del experto de la PC principal. La interfaz y el nodo solicitante permanecieron sin PyTorch; el recibo identifica al trabajador físico persistente. La identidad del cliente de prueba no aumenta el número de máquinas.

El historial se conservó cifrado con la clave en el almacén protegido real del sistema y se recuperó al reabrir la ventana. Por defecto el broker envía solo la pregunta actual. Al activar la casilla de contexto, el experto recordó el nombre ficticio «Lavanda». La prueba también verificó cancelación y ausencia del marcador sintético privado en los archivos de estado y registros del nodo. Las pruebas de IPC rechazan credenciales incorrectas, orígenes de navegador, campos desconocidos y respuestas sin recibo válido; no exportan mensajes al estado público.

Una repetición detectó la entrega pendiente del resultado de Qt después de terminar el hilo. La interfaz ahora mantiene la consulta pendiente hasta procesar el resultado. Otro ingreso vacío esperó detrás de identidades antiguas en la cola de pares; el broker prioriza una conexión acotada al experto conocido sin bloquear los controles. El ensayo final pasó ambos recorridos y la cancelación.

Pasaron 165 pruebas de regresión. Los servicios y lanzadores de ambas PCs usan el paquete actual y conservan identidades, referencia versión 6 y sus controles previos: conectividad en la PC principal y perfil equilibrado en la laptop. El experto se dejó preparado pero deshabilitado después del piloto; se corrigió su comando local para conservar el intérprete del entorno de cálculo. No se sustituyó ningún banco personal ni se entrenó con conversación privada.

Esta integración comprueba consulta remota y persistencia de conversación; no completa Matriarca, Dolphin o la preparación automática de lenguaje. La descarga pública sigue en 0.4.0a2. Faltan base y adaptación replicadas por LSP, motor de lenguaje instalable, entrenamiento distribuido de esa familia y transferencia evaluada entre tamaños. El goal mantiene las ocho etapas y sus condiciones de aceptación.

El [CI del chat/app 526ca83](https://github.com/toxxy/LixySwarm/actions/runs/38612544786) completó Linux, Windows y macOS: controles, cifrado con fixture, autenticación y cancelación del puente local, empaquetado y ejecutable con preparación del motor CPU público. Las consultas reales al experto GPU y el almacén protegido real pertenecen al ensayo Linux de las PCs. El VPS adquirió el código por Git y mantuvo relay/API activos, sin incorporar bases, adaptaciones ni motores de lenguaje.

### Registro de continuidad

`experiments/observe_internet.py` observa la API pública del Explorer cada 20 segundos para las tres identidades físicas explícitas. Conserva muestras y resumen fuera de Dropbox. Solo acumula una ventana saludable cuando todas tienen telemetría firmada, activa y reciente, con secuencia y tiempo sin retrocesos. Un intervalo sin observación mayor de 60 segundos, una identidad duplicada o un estado inválido reinician la ventana. Dos pruebas verifican esos rechazos y el requisito de duración completa.

El objetivo del registro es reunir una ventana de 24 horas; la ejecución tiene un límite de 48 horas. No sustituye la prueba del segundo relay independiente ni mide calidad del modelo. La condición de continuidad permanece pendiente hasta revisar las muestras completas y el resumen final.

Tras la comprobación corta del CLI sobre la API pública, se inició el registro largo el 2026-10-14 como servicio de usuario separado del participante. Su revisión de origen es `c08321c`; conserva evidencia fuera del repositorio y no altera los controles de aportación. La observación inicial vio las tres máquinas saludables. Ese inicio no acredita todavía las 24 horas.

### Familias de lenguaje firmadas y copia por Internet

El catálogo de `src/release/language_catalog.py` publica dos genesis de expertos: `a43dba6c…` para la base menor y `3095364a…` para la mayor. Vinculan las adaptaciones ya evaluadas, dependencias completas de upstream, tokenizer, licencia y evaluación. Las bases mantienen sus hashes originales y formato safetensors; el proceso de publicación rechaza modelos o adaptaciones diferentes. La firma usa el publicador existente del piloto y los clientes fijan los genesis concretos.

El almacén de familias se separa del de referencia de bytes, con objetos de hasta 4 GiB y cuota propia máxima de 8 GiB. Ambas áreas, archivos parciales y motores cuentan en el permiso total de disco. Las familias comparten el servicio y la identidad física persistente; ninguna ruta nueva permite ejecutar comandos enviados por pares. La inferencia y los permisos de GPU conservan su configuración anterior.

El tráfico de objetos continúa limitado por conexión y leases. El coordinador permite hasta 600 ofertas por minuto por par, conserva cuatro ofertas simultáneas como máximo y limita la caché de resultados a 32 MiB, además del número de entradas. El límite anterior de 180 ofertas hacía insuficiente el transporte de pesos grandes. Los clientes comprueban los tamaños firmados antes de recibir y verifican chunks y el SHA-256 completo antes de conservar cada objeto.

Ambas PCs instalaron el nodo 0.4.0a5. La PC principal conserva CPU 5 %, GPU cero y RAM 0,5 GiB; el piloto aumentó temporalmente solo disco a 8 GiB para servir 5.604.263.444 bytes públicos. La laptop conserva su perfil equilibrado y cuota de 10 GiB. La consulta sobre la cuota permanente sigue pendiente de la preferencia del operador.

El cliente de copia en la laptop está activo y recibe la base menor a través del relay público. Se observaron 188.067.860 bytes conservados o parciales durante la transferencia. Su identidad pertenece a un solicitante de ensayo, no a otro trabajador físico. La comprobación de conexión rechazó rutas privadas; los pesos no viajaron por Dropbox, Git ni SSH. El primer descubrimiento requirió reintento y queda registrado. La copia no se declara completa hasta verificar todos los objetos y cargar la familia adquirida.

Pasaron 177 pruebas de regresión, incluidas firmas, pins, rechazo de nombres, tamaños previos, aislamiento de namespace, transporte con fixture y presupuesto de caché. El ejecutable Linux pasó conexión y controles con estado vacío. El registro de continuidad sigue vivo, con las tres identidades físicas explícitas; aún no reúne 24 horas. Continúan pendientes las réplicas completas, motor de lenguaje instalable, entrenamiento distribuido de esas adaptaciones, transferencia entre tamaños y las demás condiciones del plan.

El [CI del participante af7fac5](https://github.com/toxxy/LixySwarm/actions/runs/38619028608) y el [CI del motor CPU](https://github.com/toxxy/LixySwarm/actions/runs/38619028564) completaron Linux, Windows y macOS. Verificaron transporte ligero, catálogo, controles y empaquetado; no ejecutaron modelos de lenguaje en esas tres plataformas. La copia real continuó con el mismo proceso, alcanzando 341.422.100 bytes parciales o conservados. El VPS adquirió el código por Git; relay/API permanecieron activos y no recibieron bases, adaptaciones ni motores de lenguaje.

### Primera familia adquirida y motores independientes

La [copia de la familia menor](experiments/reports/language_family_acquisition_20261014.json) terminó en la laptop: 1.519.351.332 bytes, todos los objetos verificados y el mismo genesis y adaptador. La materialización crea archivos con nombres upstream cerrados, comprueba contenido y conserva el experto y los permisos existentes. Los hardlinks reducen duplicación local cuando están disponibles y la cuota cuenta cada inode una vez. La laptop cargó la copia adquirida y produjo una traducción; ese primer cálculo usó aún el intérprete de desarrollo. La copia mayor se inició con otro directorio e identidad de solicitante, sin representar otro trabajador físico.

Los [motores de lenguaje](engines/LANGUAGE.md) incluyen intérprete y dependencias separados de la app. El [ensayo nativo](experiments/reports/language_native_20261014.json) verificó ejecutables CPU/CUDA, un contenedor Ubuntu 24.04 sin Python y una petición supervisada CUDA. La instalación CUDA local verificó tres partes y archivo completo, todos los archivos y aliases internos. Preparar un motor deja un registro de su ejecutable sin habilitar cálculo o sustituir el experto activo.

La petición supervisada respondió «Madrid» a una pregunta reformulada sobre Francia. El mismo prompt y adaptador reprodujeron el resultado con el intérprete de desarrollo. La prueba confirma operación del programa y revela una limitación semántica adicional: el criterio original 12/12 no cubría ese prompt. La familia pequeña no se habilita para conversación general; se requieren evaluación ampliada y conservación más fuerte antes de promociones futuras.

Pasaron 183 pruebas. El [CI de lenguaje CPU 214a67d](https://github.com/toxxy/LixySwarm/actions/runs/38627097359) completó arquitectura y adaptadores sintéticos, empaquetado y ejecutable en Linux, Windows y macOS. Los [motores de referencia](https://github.com/toxxy/LixySwarm/actions/runs/38627097381) y la [app](https://github.com/toxxy/LixySwarm/actions/runs/38627097357) también pasaron en las tres plataformas. Estos fixtures no prueban calidad general ni aprendizaje distribuido nuevo.

La publicación `language-engine-0.4.0a1` contiene el motor CPU Linux, hashes y tres partes CUDA verificadas localmente. El catálogo fija SHA-256 de cada parte, archivo unido y manifiesto; el receptor debe comprobarlos antes de ejecutar. La prueba de instalación pública CPU en la laptop está activa. La copia de base mayor también conserva su proceso; el registro de continuidad sigue pendiente de completar su ventana de 24 horas. La preparación de lenguaje desde los controles de la app, más plataformas y las demás condiciones del plan siguen abiertas.

La [prueba pública del motor CPU](experiments/reports/language_public_worker_20261014.json) terminó con éxito: la laptop descargó y verificó el paquete por HTTPS sin credencial, y su ejecutable cargó la familia menor adquirida por LSP y produjo la traducción esperada. La inferencia no usó el intérprete de desarrollo. Esto verifica la preparación subyacente y el uso de una réplica; la activación desde la interfaz y la calidad general conservan sus condiciones pendientes.

### Chat disponible y preservación factual del 2026-10-14

El operador autorizó usar todo el potencial del escritorio y señaló que está usando la laptop. El escritorio conserva el perfil máximo, con CPU y GPU al 100 % como presupuesto autorizado y un trabajo GPU simultáneo. La laptop conserva CPU/GPU al 25 %, RAM de 2 GiB y sus cuotas previas; este tramo no ejecuta entrenamiento de lenguaje en ella. La cuota temporal de disco del escritorio permanece en 8 GiB durante la réplica, sin deducir una elección permanente de almacenamiento.

El experto mayor queda activo en el escritorio. La [consulta desde el servicio productivo de la laptop](experiments/reports/chat_desktop_enabled_20261014.json) obtuvo una respuesta por LSP público en 5,132 segundos, con recibo del nodo físico `c45094…`. Las dos apps instaladas se actualizaron a 0.4.0a6. El [ensayo de interfaz en la laptop](experiments/reports/chat_app_a6_20261014.json) conectó en 0,77 segundos y comprobó respuesta remota, contexto voluntario, historial cifrado conservado al reabrir y cancelación. El cliente no cargó PyTorch ni Transformers. La página web de chat del VPS todavía usa su integración antigua; esta disponibilidad corresponde a la app de las PCs.

Una evaluación fija añadió cuarenta preguntas factuales con paráfrasis que no aparecen literalmente en el entrenamiento. El [experto menor](experiments/reports/language_paraphrase_06b_20261014.json) pasó de 32/40 a 31/40 y perdió cuatro respuestas correctas anteriores: la adaptación queda rechazada por este gate y no se selecciona automáticamente para chat. El [experto mayor](experiments/reports/language_paraphrase_17b_20261014.json) pasó de 39/40 a 40/40 sin regresiones. Es evidencia acotada de preservación factual; no acredita calidad general. Los paquetes ya compartidos conservan su identidad reproducible y carácter experimental. La preparación automática deberá respetar este rechazo, además del esquema y firma de sus archivos.

El servicio ligero incorpora la operación tipada `language.gradient.v1`: únicamente corpus público fijado, versión exacta, rangos acotados y solicitantes conocidos, con permisos GPU/RAM supervisados y revocación también en caché. El [piloto del escritorio](experiments/reports/language_gradient_desktop_20261014.json) procesó cuatro ejemplos, trece tokens de respuesta y devolvió 4.599.864 bytes de gradientes LoRA en 14,403 segundos. Se verificaron recibo físico, SHA-256, nombres, formas, float32, valores finitos y norma 0,22677. La primera petición concurrente fue rechazada por el límite de un trabajo GPU mientras el chat estaba activo; la repetición se hizo después de terminar las consultas. No se promovió ningún modelo. El piloto prueba una contribución de una sola PC; el aprendizaje de lenguaje entre dos PCs y la evaluación de una actualización agregada siguen pendientes.

Pasaron 190 pruebas de transporte, permisos, chat, artefactos, modelos, procesos y aprendizaje. La observación de continuidad y la réplica de la familia mayor mantienen sus procesos y archivos reanudables. Sus gates siguen abiertos hasta registrar el resultado final.

La actualización del escritorio a 0.4.0a6 produjo una muestra de estado desconectado durante el reinicio del servicio. El observador conservó sus archivos y reinició correctamente la ventana de salud consecutiva; el tramo anterior llegó a 8.036,5 segundos y no cumple 24 horas. No se reemplazó el proceso de observación ni se ocultó esa interrupción.
