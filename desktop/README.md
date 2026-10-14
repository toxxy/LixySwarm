# App de LixySwarm

Vista previa `0.4.0a1` del participante instalable. Incluye Estado, Chat, Recursos y Actividad. La conexión, los permisos locales, la pausa y la reconexión tienen un proceso de red separado de la interfaz. El chat permanece en preparación hasta integrar el modelo y la inferencia distribuida.

El ejecutable incluye Python y los componentes de interfaz y transporte. La instalación por usuario conserva la identidad y el estado fuera del paquete. En Linux crea una entrada en el menú de aplicaciones; Windows y macOS tienen sus propios adaptadores de instalación.

## Prueba de empaquetado

Se usa Qt Widgets mediante PySide6 Essentials. Su [distribución oficial](https://doc.qt.io/qtforpython-6/package_details.html) separa los módulos esenciales de los adicionales. La interfaz no incorpora un navegador ni un motor de modelos. [PyInstaller](https://pyinstaller.org/en/stable/operating-mode.html) incluye el intérprete y requiere construir en cada sistema operativo; el workflow `Desktop packages` utiliza runners estándar de Linux, Windows y macOS.

El build se ejecuta en un entorno aislado con las dependencias de `desktop/requirements.txt` y PyInstaller 6.22.3. Se excluyen PyTorch, NumPy, tokenizers y órganos de entrenamiento del paquete de conectividad.

```bash
python -m pip install -r desktop/requirements.txt PyInstaller==6.22.3
python desktop/build.py --output /directorio/local/de/build
```

Cada paquete tiene una prueba del ejecutable con directorio de estado nuevo: conexión cifrada, confirmación de límites por el proceso real, pausa conservando conexión, desconexión y reconexión con la misma identidad. El reporte y una captura se producen con `--self-test --report reporte.json --screenshot vista.png`. `--internet` usa el bootstrap público y comprueba direcciones públicas de transporte. Las pruebas ordinarias usan un seed temporal en loopback.

Las licencias incluidas por las dependencias se copian al paquete junto a [THIRD_PARTY.md](THIRD_PARTY.md). Los binarios compartidos permanecen separados. Los checksums describen los archivos del paquete; la firma de distribución y las validaciones de instalación pública conservan sus gates pendientes.

## Estado y permisos

La interfaz escribe `controls.json` en el directorio local del nodo. El proceso valida esquema, presupuesto y consentimiento, confirma lo recibido en sus diagnósticos y publica solo el estado público permitido. Pausar conserva conectividad; desconectar cierra las sesiones. Una configuración inválida desactiva la conexión hasta corregirla.

Los presupuestos se preparan para los trabajadores. Esta vista previa mantiene entrenamiento e inferencia deshabilitados, y los contadores de cálculo permanecen en cero. La comprobación de límites durante trabajos reales corresponde a la integración de esos trabajadores.

La clave de cifrado personal se genera en el almacén protegido del sistema. Si falta o está bloqueado, la memoria personal conserva estado pendiente. No hay clave alternativa en archivo plano. La pérdida de una clave existente no genera una sustituta que inutilice el contenido cifrado. La integración de Matriarca y conversación persistente continúa en las etapas de aprendizaje y memoria.

La instalación por usuario guarda cada versión del paquete identificada por su contenido y actualiza el lanzador. Conserva las versiones previas y mantiene el estado personal fuera de ellas. El arranque automático se integra con el servicio de referencia de Linux; la configuración equivalente por plataforma sigue pendiente.
