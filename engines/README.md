# Motores de cálculo separados

El nodo y la app se conectan sin cargar PyTorch. Este paquete CPU incluye su intérprete, el runtime de tensores y las operaciones del enjambre de referencia. Ejecuta un trabajo supervisado por stdin y devuelve JSON; no inicia una identidad ni escucha conexiones. El supervisor del nodo conserva permisos, cancelación, plazo y reserva de RAM.

El motor actual sirve inferencia de bytes, gradientes y evaluación de candidatas del modelo de referencia. No ofrece conversación ni ejecuta modelos o código arbitrario de los peers. La instalación del motor no concede recursos ni habilita entrenamiento de solicitantes nuevos.

El motor separado de lenguaje incorpora una política opcional de lotes de hasta 64 ejemplos públicos distintos. Se construye con `python engines/build_language.py --device cuda --batch-policy --output /directorio/local` y se verifica con el perfil local `language-cuda-batches`. Amplía las operaciones de lenguaje con `language.gradient.v3` y `language.candidate.v2`; los perfiles existentes conservan sus listas. Los trabajadores deben habilitar expresamente `batching_enabled`; instalar el motor no concede ese permiso. Las nuevas evaluaciones necesitan A14, conservan los gates históricos y exigen las mismas condiciones de calidad. La verificación física y la preparación desde una instalación vacía siguen siendo comprobaciones aparte.

La política matemática A18 se construye con `python engines/build_language.py --device cuda --math-policy --output /directorio/local` y se verifica mediante `language-cuda-math`. Incluye las políticas anteriores y añade `language.gradient.v5` y `language.candidate.v4`. Requiere también `math_curriculum_enabled`, deshabilitado al migrar configuraciones. Los datos fijados se adquieren antes del proceso tensorial mediante LSP público. El motor combina supervisión matemática y repaso de entrenamiento público, conserva los límites de precisión float16 y produce una traza completa para verificar las candidatas. Sus pruebas sintéticas no certifican calidad del modelo: la [prueba física](../experiments/reports/language_math_native_scaling_20261015.json) rechazó las cuatro candidatas. Ambos motores locales pasaron verificación de manifiesto y selftest; su instalación GPU vacía conserva las condiciones pendientes.

## Construcción y verificación

Crear un entorno separado fuera del checkout. Instalar `torch==2.12.0` desde el índice CPU oficial y las dependencias fijadas en `requirements-cpu.txt`, junto con `PyInstaller==6.22.3`. Después:

```bash
python engines/build.py --output /ruta/fuera/del/proyecto/engine-package
/ruta/fuera/del/proyecto/engine-package/dist/LixySwarmEngineCPU/LixySwarmEngineCPU --self-test --report /ruta/fuera/del/proyecto/report.json
```

El ejecutable incorpora licencias y un manifiesto con hashes de cada archivo, plataforma, arquitectura y operaciones. Su self-test ejecuta inferencia, dos gradientes y evaluación de una candidata sin firma ni promoción; utiliza fixtures de firma locales y no presenta estos procesos como máquinas de la red. `src.network.engine_package.install_package` requiere el SHA-256 confiable del manifiesto como argumento local independiente; no lo obtiene del paquete ni de un peer. Verifica origen, copia e instalación final, rechaza rutas que escapen, archivos adicionales, corrupción e incompatibilidad, y conserva la lista local de coordinadores autorizados.

El workflow `CPU engine packages` construye y comprueba Linux por defecto; permite incluir Windows y macOS mediante `all_platforms` para la validación final. Los artefactos de CI son resultados de desarrollo; todavía faltan el catálogo de descarga pública verificada y su preparación automática desde la app. La distribución de motores CUDA conserva su trabajo pendiente. El paquete CPU no anuncia CUDA, y la selección del trabajo respeta tanto el motor instalado como los recursos consentidos.
