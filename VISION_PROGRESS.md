# Avance del goal de LixySwarm

El usuario activó [PLAN_VISION.md](PLAN_VISION.md) como goal el 2026-10-13.
Las etapas se completan por sus pruebas de aceptación; una implementación parcial conserva la etapa pendiente.

| Etapa | Estado | Evidencia y trabajo restante |
|---|---|---|
| 1. Red continua | En observación | Servicios desplegados en dos PCs y VPS, estado por identidad en el Explorer y recuperación tras reinicio comprobados. Faltan operación de 24 horas y recuperación con otro relay público independiente. |
| 2. App sencilla | En implementación | Interfaz de cuatro pantallas, controles locales confirmados, paquetes probados en Linux/Windows/macOS y app instalada en ambas PCs. Faltan modelo de red, preparación de trabajadores, arranque automático por plataforma y gate de instalación limpia completo. |
| 3. Aprendizaje distribuido | En implementación | Pérdida colectiva y diversidad conectadas al gradiente; ciclo candidato con fragmentos distintos, evaluación, promoción y recuperación implementado. Falta el ensayo completo por Internet y su integración con el modelo conversacional y la app. |
| 4. Conocimiento persistente | Pendiente | Existen artefactos por contenido. Falta demostrar réplica y adquisición después de apagar al productor. |
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

Próximo tramo: modelo de referencia público, evaluación reproducible, trabajadores compatibles con las dos PCs y actualización candidata real sobre fragmentos distintos. La adquisición de versiones y la inferencia deben incorporarse después a la app.

### Decisiones del ciclo de aprendizaje

- La pérdida principal corresponde a los logits combinados que entrega el enjambre. En entrenamiento se añade una pérdida auxiliar de agentes con peso 0,25 y una penalización diferenciable de similitud de feromonas con peso 0,01 y margen 0,7. La evaluación reporta la pérdida predictiva colectiva. Estos pesos son valores iniciales configurables; su efecto de calidad y especialización debe medirse.
- El entrenamiento productivo reparte rangos sin solapamiento y combina sus gradientes por cantidad de tokens predichos. La repetición del mismo rango mantiene su función de verificación en el agregador de mediana existente. No se relajan sus requisitos de quórum.
- El primer piloto productivo fija explícitamente dos identidades de trabajadores conocidos. La aceptación abierta de desconocidos conserva los gates de validación pendientes.
- Los cambios se aplican sobre un modelo candidato separado. Se conservan pesos y optimizador al promoverlo, con informe de evaluación y recibos en una release firmada. Una regresión conserva la versión activa. Una promoción interrumpida se recupera sin volver a aplicar gradientes.
- `experiments/internet_learning.py` usa un enjambre de referencia pequeño y reproducible, con tres agentes, Dolphin y feromonas. Su corpus sintético y su evaluación quedan separados; no carga memoria personal ni historial de especialización del checkout. El ensayo inicial prueba el recorrido y la pérdida de lenguaje; el modelo conversacional sigue pendiente.

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
