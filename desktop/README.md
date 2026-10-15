# App de LixySwarm

La [descarga pública Linux `0.4.0a12`](https://github.com/toxxy/LixySwarm/releases/download/v0.4.0a12/LixySwarm-0.4.0a12-linux-x86_64.tar.gz) incluye Estado, Chat, Recursos y Actividad. La conexión, los permisos locales, la pausa y la reconexión tienen un proceso de red separado de la interfaz. El chat integra consultas al experto del piloto y conversación cifrada local, además de seguimiento de versiones evaluadas sin descargar sus pesos. Recursos prepara la versión evaluada vigente del experto GPU compatible, con reanudación y varios proveedores, y comprueba su respuesta antes de seleccionarla. Las instalaciones limpias integrales de lenguaje conservan sus condiciones pendientes. Windows y macOS quedan para la validación final.

La coordinación de lenguaje y el seguimiento automático se habilitan mediante configuración local explícita en los nodos del piloto. Actividad muestra la versión compartida y las rondas; el coordinador conserva cursor y recibos tras un reinicio, espera a las réplicas y se detiene cuando las candidatas dejan de mejorar. La selección de una mejora conserva el experto habilitado y sus permisos. Conectar o usar el chat no autoriza entrenamiento ni amplía recursos.

El código A13 añade reparto por capacidad observada entre las dos PCs conocidas. Los controles de coordinación versión 2 admiten `shards_per_cycle` de 2 a 16 y `allocation` igual a `balanced` u `observed`. Cada fragmento conserva el máximo de ocho ejemplos; la nueva configuración usa cuatro fragmentos y necesita dos observaciones verificadas por equipo antes de ponderar el reparto. Un archivo existente de versión 1 conserva dos fragmentos y reparto equilibrado al migrar. Los permisos GPU de cada nodo prevalecen sobre el plan. La [medición física](../experiments/reports/language_capacity_allocation_physical_20261014.json) distribuyó 24 ejemplos al escritorio y ocho a la laptop, pero resultó un 2,1 % más lenta que el escritorio solo. Esto aún no acredita acelerar el aprendizaje al sumar PCs. La descarga pública enlazada sigue siendo A12.

A13 también incluye transferencias comprimidas sin pérdida, negociadas con cada proveedor. Conserva los bytes y hashes de modelos y gradientes, las descargas parciales y el protocolo de los nodos anteriores. Las cuotas de disco se calculan sobre el archivo original; el presupuesto de red se aplica a los datos codificados y su margen de protocolo. En la [prueba con el servicio nativo](../experiments/reports/artifact_codec_native_provider_20261014.json), adquirir el mismo gradiente tardó una mediana de 27,105 segundos frente a 30,079 con la codificación anterior. Es una medición de transferencia; sigue pendiente demostrar aceleración del aprendizaje al sumar PCs.

A14 añade una política separada de hasta 64 ejemplos públicos distintos por gradiente, con permiso local `batching_enabled` y motores verificados del perfil `language-cuda-batches`. Conserva los gates históricos y la última cabeza compatible para clientes anteriores. En la [comparación del recorrido nativo](../experiments/reports/language_native_batch_scaling_20261014.json), los mismos 192 ejemplos tardaron 28,68 segundos en el escritorio y 39,74 entre las dos PCs; con evaluación, 46,00 y 57,06 segundos. Ambas candidatas se rechazaron y la versión aceptada permanece en 2. Los nodos actuales están en A14; la descarga pública sigue siendo A12 y conserva sus propias condiciones pendientes de preparación integral.

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

Los presupuestos se aplican a trabajadores separados del nodo. El motor CPU público ejecuta inferencia tipada del modelo de bytes. El entrenamiento de referencia se limita a las identidades conocidas del piloto. El experto de lenguaje necesita una instalación local verificada y recursos suficientes. El primer motor CUDA público admite Linux x86_64 con glibc 2.39 o posterior; las demás plataformas usan el chat remoto mientras se verifica su motor local. La aceptación de aportes de desconocidos sigue pendiente.

El botón **Conservar conocimiento público** habilita adquisición, verificación y servicio de versiones del catálogo público dentro del permiso de almacenamiento y conexión. El catálogo inicial contiene el enjambre de referencia de bytes que superó el piloto distribuido; aún no es el modelo conversacional. El servicio descarga la cadena desde un participante disponible, comprueba firmas, genesis y hashes, y conserva los pesos al pausar, desconectar o reiniciar. Usa la identidad del equipo y un almacén `public/` separado de la memoria personal. No carga PyTorch ni ejecuta un modelo al compartir archivos.

Una reducción de cuota por debajo de los archivos conservados pausa su servicio; no elimina conocimiento automáticamente. Dejar de compartir conserva los archivos y detiene nuevas transferencias. El presupuesto de transferencia se aplica a chunks codificados y puede interrumpirse; por ahora se limita además a 5 Mbps por servicio para respetar la ventana de bytes LSP. Descubrimiento global por contenido, reparación automática y distribución del experto de lenguaje siguen pendientes.

La clave de cifrado personal se genera en el almacén protegido del sistema. Si falta o está bloqueado, el chat informa que conserva la conversación temporalmente en memoria. No hay clave alternativa en archivo plano. La pérdida de una clave existente no genera una sustituta que inutilice el contenido cifrado.

La fuente A15 añade **Recuperar recuerdos de esta conversación**, desactivado inicialmente. El componente textual de Matriarca personal consolida intercambios completos y busca hasta dos fragmentos anteriores dentro del mismo presupuesto de contexto. Sus hasta 128 fragmentos se conservan cifrados aunque salgan de los 200 mensajes visibles. Sin autorización de contexto, el experto remoto recibe solamente la pregunta actual. **Vaciar recuerdos** elimina esos fragmentos y desactiva la recuperación; conserva el historial visible, que puede consolidarse otra vez al reactivarla. La memoria global, la orientación neuronal, Dolphin y las sectas mantienen su integración y evaluación pendientes. La publicación y la prueba física de esta versión se registran en VISION_PROGRESS.

## Chat del piloto

El chat se adjunta al nodo físico existente mediante una API autenticada que escucha únicamente en `127.0.0.1`. El transporte hasta el experto usa LSP cifrado por Internet. La app no carga PyTorch ni descarga los pesos para solicitar una respuesta. Si no hay un experto disponible, mantiene conexión y muestra ese estado.

Por defecto se envía solo la pregunta actual. La casilla **Enviar también el contexto de esta conversación al experto de la red** autoriza un contexto acotado de la conversación visible. El nodo no lee la memoria personal ni utiliza conversaciones para entrenamiento, feromonas o métricas públicas. El historial se conserva cifrado en `personal.enc`, con la clave en el almacén protegido del sistema; preguntas y respuestas permanecen fuera de los diagnósticos y registros del nodo.

Las respuestas deben incluir un recibo firmado del experto y corresponder exactamente a la adaptación pública del piloto. La familia mayor requiere una reserva de 5 GiB de RAM y al menos 4 GiB de GPU dentro de los límites elegidos por el operador. El piloto admite una identidad conocida; no constituye todavía descubrimiento abierto de expertos.

`--self-test --internet --chat` comprueba consultas desde el ejecutable, contexto opcional, cancelación y recuperación del historial cifrado al reabrir. Requiere el experto disponible y el almacén protegido del sistema desbloqueado. Emplea exclusivamente conversaciones sintéticas; su identidad de cliente no representa otra máquina física.

La instalación por usuario guarda cada versión del paquete identificada por su contenido y actualiza el lanzador. Conserva las versiones previas y mantiene el estado personal fuera de ellas. El arranque automático se integra con el servicio de referencia de Linux; la configuración equivalente por plataforma sigue pendiente.

## Preparación automática del motor CPU

Al guardar un perfil con CPU y al menos 2 GB de RAM, la app prepara el motor dentro del permiso de almacenamiento y conexión. En una entrada nueva también habilita el modelo público; respeta una retirada previa del permiso de compartir. La conexión a la red empieza antes de estas descargas. Recursos muestra progreso y permite reanudar; pausar o desconectar conserva el archivo parcial.

Los [motores CPU públicos](https://github.com/toxxy/LixySwarm/releases/tag/cpu-engine-0.4.0a1) se descargan por HTTPS sin cuenta. El catálogo incluido en la app fija plataforma, arquitectura, tamaño, SHA-256 del ZIP y SHA-256 del manifiesto. Se verifica todo antes de seleccionar el ejecutable. El ZIP se extrae con límites y rechazo de rutas que escapen o archivos adicionales. La preparación reserva la mitad del presupuesto de conexión y deja la otra mitad para conocimiento público.

El motor incluye su intérprete y PyTorch CPU; la app y el nodo permanecen ligeros. Una instalación CPU conserva el motor CUDA local anterior si sigue disponible. Las operaciones son las del modelo de bytes de referencia: esto no habilita conversación ni entrenamiento abierto de desconocidos. La calidad conversacional amplia y la instalación física de Windows/macOS conservan sus gates.

## Preparación del experto de lenguaje

La app puede preparar el experto aprobado del piloto al terminar el motor CPU, cuando el perfil permite CPU, 5 GiB de RAM y al menos 4 GiB de GPU por trabajo, con almacenamiento público autorizado. El botón **Preparar experto de lenguaje** permite iniciar o reanudar expresamente. Consultar al experto remoto no requiere preparar estos archivos ni conceder GPU local.

El nodo descarga las dependencias firmadas por LSP usando su identidad física. El motor se obtiene del catálogo HTTPS fijado en la app; un peer no puede elegir programas, rutas ni modelos. El experto pequeño que perdió respuestas correctas en la evaluación de preservación no se activa automáticamente. La primera preparación completa necesita alrededor de 27 GiB, incluyendo archivos temporales conservadores; el cálculo descuenta descargas parciales y paquetes ya preparados al reanudar. Archivos de modelo, motores y descargas comparten la cuota elegida, sin incluir el historial privado.

Preparar no aumenta permisos. La activación exige verificar todos los hashes y una respuesta sintética bajo una reserva real de RAM y GPU. Pausar, desconectar, cambiar límites o seleccionar otro experto impide la activación pendiente. Detener conserva los archivos verificados y cancela la intención de reanudar. Una intención guardada se recupera después de reiniciar el nodo, dentro de los permisos actuales. Las preparaciones CPU y GPU se serializan para evitar competir por la reserva de disco.
