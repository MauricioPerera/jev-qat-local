# Instalador de E2B en Windows

## Uso

1. Descarga el ZIP del instalador desde [Releases](https://github.com/MauricioPerera/jev-qat-local/releases).
2. Extrae **todos** los archivos; no ejecutes el CMD dentro del visor del ZIP.
3. Haz doble clic en **Instalar E2B.cmd** y deja la ventana abierta durante la descarga.
4. Al terminar, se abre la interfaz en el navegador predeterminado.

El ZIP es pequeño: no contiene pesos, Python ni FFmpeg. La instalación descarga
unos 4,5 GB y guarda todo en `%LOCALAPPDATA%\JevQatLocal`, sin privilegios de
administrador, cambios en el PATH global ni instalación de un servicio de Windows.
Reserva al menos 8 GB de disco. Requiere Windows 10/11 x64 y conexión a Internet;
el rendimiento y la memoria disponible dependen de tu equipo.

Usa el menú Inicio → **Jev QAT E2B → Abrir E2B** o los CMD en la carpeta instalada.
**Detener E2B** apaga esa instancia. Cerrar el navegador deja el modelo cargado.
No se inicia automáticamente al encender el equipo.

## Qué se configura

- Python 3.14.8 embebido, aislado de Python y paquetes del sistema.
- 17 wheels de ejecución con versiones, URLs y SHA-256 fijados; no instala pip
  global ni necesita compilar dependencias.
- llama.cpp b11669: selecciona el primer dispositivo Vulkan disponible. Si no lo
  encuentra, descarga el runtime de CPU y usa `device=none`, más lento.
- FFmpeg/ffprobe 9.0.2 de Gyan, disponibles solo en el entorno de esta aplicación.
- E2B QAT Q4 y su proyector desde la revisión oficial fijada, con descarga por
  rangos reanudable y verificación completa de tamaño y SHA-256.
- Puertos 18166/18167 si están libres; en otro caso busca otro par y muestra la URL.
  No detiene otras instalaciones de E2B o E4B.
- La interfaz de varias preguntas, probabilidades independientes y control
  preventivo de 4096 tokens por pregunta del adaptador existente.

Las probabilidades de E2B siguen sin calibrar. El instalador no agrega pesos de
Jev-Omni, calibraciones E4B ni resultados de evaluación privados.

El manifiesto `installer/windows/dependencies.json` fija los binarios y ruedas.
El paquete generado añade `bundle.json` con el commit y hash del código incluido.
`source.zip` es una instantánea de ese commit, no una descarga de `main` al instalar.
Se incluyen las licencias de la aplicación y se conservan las de los componentes
descargados. Consulta [NOTICE.md](../NOTICE.md).

## Opciones

Desde una consola, en la carpeta extraída:

```powershell
& '.\Instalar E2B.cmd' -InstallDir 'D:\Aplicaciones\E2B'
```

| Opción | Efecto |
|---|---|
| `-NoLaunch` | Instala herramientas y modelo sin arrancar la API |
| `-NoBrowser` | Arranca la API sin abrir el navegador |
| `-NoShortcuts` | Omite accesos del menú Inicio |
| `-DependenciesOnly` | Prepara herramientas sin descargar pesos ni arrancar la API |
| `-DownloadCache 'ruta'` | Reutiliza ZIPs y wheels verificados de esa carpeta |
| `-ModelCache 'ruta'` | Copia ambos GGUF oficiales tras verificar sus hashes |

La caché aportada es de solo lectura para el instalador. La descarga ordinaria no
necesita estas opciones. En modo `DependenciesOnly` la instalación no puede inferir;
ejecuta de nuevo sin esa opción para completarla.

La carpeta de destino debe estar vacía o pertenecer a una instalación previa de
este producto. Un bloqueo impide dos instalaciones simultáneas. Las repeticiones
conservan los puertos y reutilizan componentes completados. Detén la aplicación
antes de instalar una versión distinta. No hay actualización automática.

## Diagnóstico y retirada

Un fallo de hash detiene la instalación antes de ejecutar o extraer ese archivo.
Si una descarga del modelo se interrumpe, vuelve a ejecutar el instalador. No
desactiva TLS ni instala controladores de GPU. Si Vulkan no se detecta, revisa el
controlador si deseas aceleración; la alternativa de CPU sigue disponible.

El estado sin credenciales del instalador está en `install.json`. Los registros del
servicio están en `app/integration/jev-qat-e2b/service.log` y `backend.log`. El archivo
`service.json` contiene un token administrativo local: no lo publiques.

El instalador consiste en scripts legibles y no tiene firma Authenticode. Se
publica un archivo `.zip.sha256` junto al ZIP para comprobar su integridad:

```powershell
Get-FileHash .\JevQat-E2B-Windows-0.2.0.zip -Algorithm SHA256
Get-Content .\JevQat-E2B-Windows-0.2.0.zip.sha256
```

Para retirarlo, usa **Detener E2B**, guarda los archivos que quieras conservar y
elimina la carpeta instalada y sus accesos del menú Inicio. No incluye un borrado
automático de modelos ni de cargas del usuario.

## Construcción y pruebas

Desde un checkout con Git y Python 3.11+:

```powershell
python scripts/build_windows_installer.py --ref HEAD --output dist/windows
```

Solo empaqueta código de un commit; rechaza directorios de datos o archivos
inesperados y genera ZIP + SHA-256. Usa una carpeta de salida nueva si ya existe
un paquete con la misma versión.

Las pruebas unitarias verifican hashes, caché, rutas de ZIP, enlaces y puertos.
La CI ejecuta además el bootstrap real con **Windows PowerShell 5.1**, Python
privado, las 17 dependencias y FFmpeg en una carpeta con espacios y caracteres
Unicode. Repite el instalador para comprobar su reutilización. Esa prueba no
descarga 4,34 GB de pesos ni hace inferencia: la inferencia real se comprueba
por separado en hardware local antes de publicar la versión.
