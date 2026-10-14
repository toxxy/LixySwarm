# Componentes del paquete

La app utiliza Python, PySide6/Qt y Shiboken, cryptography, psutil y keyring,
con sus dependencias de plataforma. El proceso de empaquetado incluye las
licencias distribuidas con esos componentes en el directorio `licenses`.
Los binarios compartidos de Qt permanecen separados y pueden reemplazarse.

Código del proyecto: https://github.com/toxxy/LixySwarm

Fuentes y documentación de los componentes:

- Qt for Python: https://code.qt.io/cgit/pyside/pyside-setup.git/ y https://doc.qt.io/qtforpython-6/package_details.html
- Qt: https://code.qt.io/cgit/qt/ y https://www.qt.io/licensing/open-source-lgpl-obligations
- Python: https://www.python.org/downloads/source/
- cryptography: https://github.com/pyca/cryptography
- psutil: https://github.com/giampaolo/psutil
- keyring: https://github.com/jaraco/keyring

PyInstaller se usa para construir el ejecutable con el intérprete incluido.
Documentación de empaquetado: https://pyinstaller.org/en/stable/operating-mode.html
