# App de LixySwarm

Vista previa `0.4.0a2` del participante instalable. Incluye Estado, Chat, Recursos y Actividad. La conexión, los permisos locales, la pausa y la reconexión tienen un proceso de red separado de la interfaz. El chat permanece en preparación hasta integrar el modelo y la inferencia distribuida.

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

El botón **Conservar conocimiento público** habilita adquisición, verificación y servicio de versiones del catálogo público dentro del permiso de almacenamiento y conexión. El catálogo inicial contiene el enjambre de referencia de bytes que superó el piloto distribuido; aún no es el modelo conversacional. El servicio descarga la cadena desde un participante disponible, comprueba firmas, genesis y hashes, y conserva los pesos al pausar, desconectar o reiniciar. Usa la identidad del equipo y un almacén `public/` separado de la memoria personal. No carga PyTorch ni ejecuta un modelo al compartir archivos.

Una reducción de cuota por debajo de los archivos conservados pausa su servicio; no elimina conocimiento automáticamente. Dejar de compartir conserva los archivos y detiene nuevas transferencias. El presupuesto de transferencia se aplica a chunks codificados y puede interrumpirse; por ahora se limita además a 5 Mbps por servicio para respetar la ventana de bytes LSP. Descubrimiento global por contenido, reparación automática y modelo de chat siguen pendientes.

La clave de cifrado personal se genera en el almacén protegido del sistema. Si falta o está bloqueado, la memoria personal conserva estado pendiente. No hay clave alternativa en archivo plano. La pérdida de una clave existente no genera una sustituta que inutilice el contenido cifrado. La integración de Matriarca y conversación persistente continúa en las etapas de aprendizaje y memoria.

La instalación por usuario guarda cada versión del paquete identificada por su contenido y actualiza el lanzador. Conserva las versiones previas y mantiene el estado personal fuera de ellas. El arranque automático se integra con el servicio de referencia de Linux; la configuración equivalente por plataforma sigue pendiente.

## Preparación automática del motor CPU

Al guardar un perfil con CPU y al menos 2 GB de RAM, la app prepara el motor dentro del permiso de almacenamiento y conexión. En una entrada nueva también habilita el modelo público; respeta una retirada previa del permiso de compartir. La conexión a la red empieza antes de estas descargas. Recursos muestra progreso y permite reanudar; pausar o desconectar conserva el archivo parcial.

Los [motores CPU públicos](https://github.com/toxxy/LixySwarm/releases/tag/cpu-engine-0.4.0a1) se descargan por HTTPS sin cuenta. El catálogo incluido en la app fija plataforma, arquitectura, tamaño, SHA-256 del ZIP y SHA-256 del manifiesto. Se verifica todo antes de seleccionar el ejecutable. El ZIP se extrae con límites y rechazo de rutas que escapen o archivos adicionales. La preparación reserva la mitad del presupuesto de conexión y deja la otra mitad para conocimiento público.

El motor incluye su intérprete y PyTorch CPU; la app y el nodo permanecen ligeros. Una instalación CPU conserva el motor CUDA local anterior si sigue disponible. Las operaciones son las del modelo de bytes de referencia: esto no habilita conversación ni entrenamiento abierto de desconocidos. La distribución automática CUDA, la calidad conversacional y la instalación física de Windows/macOS conservan sus gates.
