# Avance del goal de LixySwarm

El usuario activó [PLAN_VISION.md](PLAN_VISION.md) como goal el 2026-10-13.
Las etapas se completan por sus pruebas de aceptación; una implementación parcial conserva la etapa pendiente.

| Etapa | Estado | Evidencia y trabajo restante |
|---|---|---|
| 1. Red continua | En observación | Servicios desplegados en dos PCs y VPS, estado por identidad en el Explorer y recuperación tras reinicio comprobados. Faltan operación de 24 horas y recuperación con otro relay público independiente. |
| 2. App sencilla | En implementación | Cuatro pantallas, controles confirmados, catálogo y adquisición de la referencia pública desde el ejecutable; paquetes verificados en tres sistemas y app instalada en ambas PCs. Faltan motores de cálculo, modelo conversacional, arranque automático por plataforma y gate completo de instalación limpia. |
| 3. Aprendizaje distribuido | Piloto verificado; integración pendiente | Dos GPUs procesaron ocho fragmentos por Internet: cuatro promociones y pérdida 5,56086 → 5,45485. Ambas adquirieron y usaron el modelo final. Faltan recuperación de trabajo incompleto, modelo conversacional e integración con la app. |
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
