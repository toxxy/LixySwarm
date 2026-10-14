# Avance del goal de LixySwarm

El usuario activó [PLAN_VISION.md](PLAN_VISION.md) como goal el 2026-10-13.
Las etapas se completan por sus pruebas de aceptación; una implementación parcial conserva la etapa pendiente.

| Etapa | Estado | Evidencia y trabajo restante |
|---|---|---|
| 1. Red continua | En observación | Servicios desplegados en dos PCs y VPS, estado por identidad en el Explorer y recuperación tras reinicio comprobados. Faltan operación de 24 horas y recuperación con otro relay público independiente. |
| 2. App sencilla | En implementación | Cuatro pantallas, controles confirmados, catálogo y adquisición de la referencia pública desde el ejecutable; paquetes verificados en tres sistemas y app instalada en ambas PCs. Faltan motores de cálculo, modelo conversacional, arranque automático por plataforma y gate completo de instalación limpia. |
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
