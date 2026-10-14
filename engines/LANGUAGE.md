# Motores de lenguaje

Los motores CPU y CUDA incluyen su intérprete, PyTorch, Transformers y la arquitectura Qwen3. Los pesos, tokenizer y adaptaciones se adquieren por LSP y se verifican por separado. El proceso no importa identidad de transporte, publicador, Matriarca ni almacén personal. Las operaciones normales usan archivos locales fijados por hash y modo offline; las restricciones de filesystem del proceso siguen pendientes.

`build_language.py --device cpu|cuda --output /directorio/de/build` requiere un entorno con el runtime correspondiente. El paquete mantiene licencias, manifiesto y hash de cada archivo. `language_main.py --self-test` comprueba arquitectura y adaptador con un fixture pequeño; no mide la calidad del modelo público.

El perfil de instalación se elige en la app, junto con el digest de confianza. El paquete descargado y los pares LSP no eligen ese perfil. Preparar un motor registra su ejecutable verificado sin cambiar el experto activo ni conceder permisos de cálculo. El perfil CPU de referencia conserva sus límites y operaciones anteriores.

El motor CUDA supera el tamaño de un archivo de distribución; se divide en tres partes de hasta 1 GiB. El receptor reanuda y verifica cada parte y después el archivo completo, antes de extraer. Los aliases de bibliotecas permanecen dentro del paquete y deben resolver a archivos regulares verificados. Una ruta fuera del paquete se rechaza incluso con un archivo de distribución fijado.

Los primeros paquetes locales corresponden a Linux x86_64 Ubuntu 24.04. El motor CPU pasó un contenedor limpio sin Python. Hay CI de CPU para Linux, Windows y macOS; eso no acredita inferencia real en las tres plataformas ni instalación física. CUDA continúa con el ensayo de las PCs Linux.

La [publicación experimental](https://github.com/toxxy/LixySwarm/releases/tag/language-engine-0.4.0a1) ofrece el archivo CPU y las tres partes CUDA de Linux Ubuntu 24.04. `engine_catalog.language_engine` fija sus tamaños, hashes de archivos y manifiestos; descarta sistemas Linux anteriores al entorno validado. La descarga pública y la integración del recorrido de preparación en la app se verifican por separado.

La prueba del ejecutable CUDA con la familia pequeña obtuvo una respuesta factual incorrecta al reformular una pregunta. El intérprete de desarrollo reprodujo el fallo. El empaquetado funciona, pero no habilita conversación general ni sustituye los criterios pendientes de evaluación, réplica y aprendizaje distribuido.
