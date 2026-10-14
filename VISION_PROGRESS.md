# Avance del goal de LixySwarm

El usuario activó [PLAN_VISION.md](PLAN_VISION.md) como goal el 2026-10-13.
Las etapas se completan por sus pruebas de aceptación; una implementación parcial conserva la etapa pendiente.

| Etapa | Estado | Evidencia y trabajo restante |
|---|---|---|
| 1. Red continua | En observación | Servicios desplegados en dos PCs y VPS, estado por identidad en el Explorer y recuperación tras reinicio comprobados. Faltan operación de 24 horas y recuperación con otro relay público independiente. |
| 2. App sencilla | En implementación | Interfaz de cuatro pantallas, controles locales confirmados y paquete Linux probado por Internet e instalado por usuario. Faltan resultados de los paquetes Windows/macOS, modelo de red, preparación de trabajadores y gate de instalación limpia completo. |
| 3. Aprendizaje distribuido | Pendiente | Existen gradientes remotos. Falta completar actualización, evaluación y promoción con fragmentos distintos. |
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

La elección de Qt se mantiene sujeta a los resultados de empaquetado en las tres plataformas. Los presupuestos de CPU/GPU preparados en la interfaz todavía deben comprobarse con trabajadores de cálculo reales. No se marca la etapa como completada por los controles o el paquete Linux.

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
