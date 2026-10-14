# Motores de cálculo separados

El nodo y la app se conectan sin cargar PyTorch. Este paquete CPU incluye su intérprete, el runtime de tensores y las operaciones del enjambre de referencia. Ejecuta un trabajo supervisado por stdin y devuelve JSON; no inicia una identidad ni escucha conexiones. El supervisor del nodo conserva permisos, cancelación, plazo y reserva de RAM.

El motor actual sirve inferencia de bytes, gradientes y evaluación de candidatas del modelo de referencia. No ofrece conversación ni ejecuta modelos o código arbitrario de los peers. La instalación del motor no concede recursos ni habilita entrenamiento de solicitantes nuevos.

## Construcción y verificación

Crear un entorno separado fuera del checkout. Instalar `torch==2.12.0` desde el índice CPU oficial y las dependencias fijadas en `requirements-cpu.txt`, junto con `PyInstaller==6.22.3`. Después:

```bash
python engines/build.py --output /ruta/fuera/del/proyecto/engine-package
/ruta/fuera/del/proyecto/engine-package/dist/LixySwarmEngineCPU/LixySwarmEngineCPU --self-test --report /ruta/fuera/del/proyecto/report.json
```

El ejecutable incorpora licencias y un manifiesto con hashes de cada archivo, plataforma, arquitectura y operaciones. `src.network.engine_package.install_package` requiere el SHA-256 confiable del manifiesto como argumento local independiente; no lo obtiene del paquete ni de un peer. Verifica origen, copia e instalación final, rechaza rutas que escapen, archivos adicionales, corrupción e incompatibilidad, y conserva la lista local de coordinadores autorizados.

El workflow `CPU engine packages` construye y comprueba Linux, Windows y macOS por separado. Los artefactos de CI son resultados de desarrollo; todavía faltan el catálogo de descarga pública verificada y su preparación automática desde la app. La distribución de motores CUDA conserva su trabajo pendiente. El paquete CPU no anuncia CUDA, y la selección del trabajo respeta tanto el motor instalado como los recursos consentidos.
