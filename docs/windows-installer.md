# Instalador de E2B en Windows

## Uso

1. Descarga **JevQat-E2B-Setup-0.3.0.exe** desde [Releases](https://github.com/MauricioPerera/jev-qat-local/releases).
2. Haz doble clic y deja la ventana de progreso abierta durante la descarga.
3. Al terminar, se abre la interfaz en el navegador predeterminado.

El EXE contiene el instalador y el código de una revisión fija; no contiene pesos,
Python ni FFmpeg. Extrae sus archivos temporales automáticamente y usa PowerShell
5.1 de Windows. Muestra el avance y guarda un registro de instalación.

El ejecutable es x64 y utiliza .NET Framework del sistema, disponible en Windows
10/11. No requiere un SDK de .NET ni herramientas de desarrollo para ejecutarlo.
La instalación descarga
unos 4,5 GB y guarda todo en `%LOCALAPPDATA%\JevQatLocal`, sin privilegios de
administrador, cambios en el PATH global ni instalación de un servicio de Windows.
Reserva al menos 8 GB de disco. Requiere Windows 10/11 x64 y conexión a Internet;
el rendimiento y la memoria disponible dependen de tu equipo.

También se publica un ZIP alternativo: extrae todos sus archivos y abre
`Instalar E2B.cmd`. No ejecutes ese CMD dentro del visor del ZIP.

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

Desde una consola, en la carpeta de la descarga:

```powershell
& '.\JevQat-E2B-Setup-0.3.0.exe' -InstallDir 'D:\Aplicaciones\E2B'
```

| Opción | Efecto |
|---|---|
| `-NoLaunch` | Instala herramientas y modelo sin arrancar la API |
| `-NoBrowser` | Arranca la API sin abrir el navegador |
| `-NoShortcuts` | Omite accesos del menú Inicio |
| `-DependenciesOnly` | Prepara herramientas sin descargar pesos ni arrancar la API |
| `-DownloadCache 'ruta'` | Reutiliza ZIPs y wheels verificados de esa carpeta |
| `-ModelCache 'ruta'` | Copia ambos GGUF oficiales tras verificar sus hashes |
| `--verify` | Solo en EXE: verifica el paquete incorporado sin instalar ni descargar |
| `--non-interactive` | Solo en EXE: no espera una tecla en caso de error, para automatización |

La caché aportada es de solo lectura para el instalador. La descarga ordinaria no
necesita estas opciones. En modo `DependenciesOnly` la instalación no puede inferir;
ejecuta de nuevo sin esa opción para completarla.

La carpeta de destino debe estar vacía o pertenecer a una instalación previa de
este producto. Un bloqueo impide dos instalaciones simultáneas. Las repeticiones
conservan los puertos y reutilizan componentes completados. Al ejecutar una versión
distinta, detiene de forma controlada únicamente la instancia de ese destino antes
de reemplazar su código. Rechaza un estado que pertenezca a otra carpeta. No busca
ni descarga actualizaciones por su cuenta.

`Ctrl+C` o cerrar el EXE durante la instalación termina sus propios procesos de
instalación. Los rangos completos del modelo se conservan para reanudar. Si la
instalación termina bien, el servicio queda funcionando después de cerrar el EXE.
No termina otras instalaciones de E2B o E4B.

## Diagnóstico y retirada

Un fallo de hash detiene la instalación antes de ejecutar o extraer ese archivo.
Si una descarga del modelo se interrumpe, vuelve a ejecutar el instalador. No
desactiva TLS ni instala controladores de GPU. Si Vulkan no se detecta, revisa el
controlador si deseas aceleración; la alternativa de CPU sigue disponible.

El estado sin credenciales del instalador está en `install.json`. Los registros del
servicio están en `app/integration/jev-qat-e2b/service.log` y `backend.log`. El archivo
`service.json` contiene un token administrativo local: no lo publiques.

El EXE no tiene firma Authenticode. Se publica un archivo `.exe.sha256` junto al
ejecutable para comprobar su integridad; el ZIP también tiene su propia suma:

```powershell
Get-FileHash .\JevQat-E2B-Setup-0.3.0.exe -Algorithm SHA256
Get-Content .\JevQat-E2B-Setup-0.3.0.exe.sha256
```

Para retirarlo, usa **Detener E2B**, guarda los archivos que quieras conservar y
elimina la carpeta instalada y sus accesos del menú Inicio. No incluye un borrado
automático de modelos ni de cargas del usuario.

Los registros del EXE se guardan en `%LOCALAPPDATA%\JevQatLocal-Setup\logs`.
No contienen el token administrativo, pero pueden contener rutas locales. El EXE
verifica el SHA-256 de su paquete interno antes de extraerlo. La suma no sustituye
una firma de editor. Tras una salida normal retira únicamente sus archivos temporales
conocidos; una interrupción forzada puede dejar un pequeño directorio temporal.

## Construcción y pruebas

Desde un checkout con Git y Python 3.11+:

```powershell
python scripts/build_windows_installer.py --ref HEAD --output dist/windows --exe
```

Solo empaqueta código de un commit; rechaza directorios de datos o archivos
inesperados y genera ZIP, EXE x64 y sus SHA-256. La construcción del EXE necesita
Windows y `csc.exe` de .NET Framework; no instala compiladores en el equipo del
usuario. El código C# y el manifiesto `asInvoker` proceden del mismo commit que
el paquete incorporado. Usa una carpeta de salida nueva si ya existe
un paquete con la misma versión.

Las pruebas unitarias verifican hashes, caché, rutas de ZIP, enlaces y puertos.
La CI ejecuta además el **EXE real** con **Windows PowerShell 5.1**, Python
privado, las 17 dependencias y FFmpeg en una carpeta con espacios y caracteres
Unicode. Repite el instalador para comprobar su reutilización. Esa prueba no
descarga 4,34 GB de pesos ni hace inferencia. Otras pruebas ejecutan el EXE compilado
con un instalador de prueba para comprobar corrupción del paquete, argumentos
Unicode, conservación del servicio al terminar y limpieza de sus propios procesos
en error o cancelación. La inferencia real se comprueba
por separado en hardware local antes de publicar la versión.
