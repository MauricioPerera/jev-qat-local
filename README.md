# jev-qat-local

[![Tests](https://github.com/MauricioPerera/jev-qat-local/actions/workflows/tests.yml/badge.svg)](https://github.com/MauricioPerera/jev-qat-local/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Adaptador local de decisiones con **Gemma 4 E2B / E4B QAT Q4_0**, su componente
multimodal y llama.cpp con Vulkan. Recibe texto o un archivo, una pregunta y entre
2 y 256 opciones; devuelve la opción elegida y un vector de probabilidades.

Incluye una interfaz web para hacer varias preguntas sobre el mismo archivo,
contar los tokens de cada pregunta y bloquear las que excedan **4096 tokens en E2B**.
La preparación del archivo y el prefijo nativo se pueden reutilizar entre preguntas.
Cada pregunta conserva sus propias opciones, contexto y probabilidades.

El contrato principal de `predict()` se inspira en
[Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni/blob/main/jev_omni.py).
El adaptador calcula los puntajes con Gemma; **no carga la cabeza entrenada,
LoRA ni pesos de Jev-Omni**. Su precisión y calibración no se transfieren aquí.

## Requisitos

- Windows x64, Python 3.11 o posterior; instalación y pruebas verificadas con Python 3.14.
- GPU con Vulkan y controlador compatible. Las mediciones se realizaron con Intel Arc 140T.
- llama.cpp **b11669** para Windows/Vulkan, según las instrucciones siguientes.
- FFmpeg y ffprobe en `PATH` para audio y vídeo.
- Espacio para E2B y su proyector: aproximadamente **4,34 GB**; E4B necesita otros **6,15 GB**.
- Node.js 22 o posterior solo para las pruebas de lógica de la interfaz.

El arranque y la instrumentación de recursos están preparados para Windows.
No se ha validado el arranque del servicio en Linux o macOS. Los valores de memoria
medidos se documentan en [benchmarks/scaling.md](benchmarks/scaling.md); no son un
requisito mínimo de RAM ni una garantía de rendimiento en otros equipos.

## Instalador sencillo para Windows

Descarga **JevQat-E2B-Windows-0.2.0.zip** desde
[GitHub Releases](https://github.com/MauricioPerera/jev-qat-local/releases/latest),
extrae todo su contenido y haz doble clic en **Instalar E2B.cmd**.

El instalador prepara Python privado, dependencias fijadas, llama.cpp y FFmpeg;
descarga E2B QAT Q4 y su proyector oficial, comprueba SHA-256 y abre la interfaz
cuando el servicio está listo. No requiere Git, Python instalado ni permisos de
administrador. Usa Vulkan si lo detecta y configura CPU como alternativa.

La descarga inicial es de unos **4,5 GB**. Reserva al menos **8 GB** de disco.
Instala en `%LOCALAPPDATA%\JevQatLocal` y crea accesos de **Abrir E2B** y
**Detener E2B** en el menú Inicio. Si se interrumpe la descarga del modelo, vuelve
a ejecutar el instalador para reanudarla. Cerrar el navegador no libera el modelo;
usa **Detener E2B**. No se configura inicio automático.

Consulta [la guía del instalador](docs/windows-installer.md) para elegir carpeta,
reutilizar descargas, diagnosticar errores o construir el paquete desde fuentes.

## Instalación manual de E2B

Ejecuta los comandos desde la raíz del repositorio. Se usa instalación editable:
la configuración, los modelos y el estado pertenecen a esta carpeta.

```powershell
git clone https://github.com/MauricioPerera/jev-qat-local.git
cd jev-qat-local
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Instala FFmpeg por el método que prefieras y comprueba ambos ejecutables:

```powershell
ffmpeg -version
ffprobe -version
```

Descarga el [runtime oficial b11669](https://github.com/ggml-org/llama.cpp/releases/tag/b11669)
y verifica el archivo antes de extraerlo:

```powershell
New-Item -ItemType Directory -Force runtimes | Out-Null
Invoke-WebRequest -Uri 'https://github.com/ggml-org/llama.cpp/releases/download/b11669/llama-b11669-bin-win-vulkan-x64.zip' -OutFile runtimes/llama-b11669-bin-win-vulkan-x64.zip
$runtimeHash = (Get-FileHash runtimes/llama-b11669-bin-win-vulkan-x64.zip -Algorithm SHA256).Hash
if ($runtimeHash -ne '2c07574e7bde71321cb785282b20cc821e951132149eeceeea32ddebdbf0598e') { throw 'SHA-256 del runtime incorrecto' }
Expand-Archive -LiteralPath runtimes/llama-b11669-bin-win-vulkan-x64.zip -DestinationPath runtimes/llama-b11669-vulkan
Test-Path runtimes/llama-b11669-vulkan/llama-server.exe
```

Descarga los pesos y el proyector **oficiales de E2B**, con revisión, tamaño y SHA-256
fijados en el script. La descarga admite reanudación y requiere acceso a Hugging Face:

```powershell
python scripts/download_gemma_e2b_qat.py
python -m jev_qat.e2b start
python -m jev_qat.e2b status
python -m jev_qat.e2b predict --input examples/decision-text.json
```

Abre **http://127.0.0.1:18167/** para usar la interfaz, o
http://127.0.0.1:18167/docs para consultar la API. Añade preguntas, escribe opciones
(una por línea), carga un archivo si corresponde y pulsa **Comprobar tokens**.
Solo las preguntas verificadas que caben en el contexto pueden enviarse a inferir.
Cambiar una pregunta invalida su comprobación; cambiar el texto o archivo compartido
invalida las comprobaciones de todas las preguntas.

Para cerrar el servicio y liberar sus recursos:

```powershell
python -m jev_qat.e2b stop
```

No se instala un servicio de inicio automático. Los puertos y rutas se configuran
con rutas relativas al checkout en [jev_qat_e2b_config.json](jev_qat_e2b_config.json).
`device` usa `Vulkan0` por defecto; ajústalo a tu equipo. Mantén el runtime fijado:
el conteo multimodal depende de su endpoint nativo de tokens.

## Entrada y respuesta

```python
from jev_qat.e2b import load_jev_omni_e2b_qat

with load_jev_omni_e2b_qat() as adapter:
    request = {
        "state": "La reunión empieza a las 10:00. Ahora son las 09:00.",
        "question": "¿Ya empezó la reunión?",
        "options": ["Sí", "No"],
    }
    budget = adapter.count_tokens(**request)
    print(budget["input_tokens"], budget["available_tokens"])
    if budget["fits_context"]:
        result = adapter.predict(**request)
        print(result["prediction"], result["probabilities"])
```

Los campos principales son `prediction`, `prediction_index` (desde cero),
`confidence` y `probabilities`, un mapa de las opciones proporcionadas. `metadata`
identifica modelo, uso de tokens, presupuesto y preparación del archivo.

Cada opción se asigna a una etiqueta verificada de un solo token. Las probabilidades
se obtienen normalizando sus puntajes; no se extraen porcentajes de texto generado.
`confidence` es el mayor puntaje condicionado a las opciones, **no una probabilidad
calibrada de acertar**. E2B devuelve `metadata.calibrated=false`. Cambiar las opciones
o su orden puede cambiar la respuesta. La tarea es elegir una opción; no produce
una explicación libre ni una transcripción completa.

Para medios, añade `media="media/example.jpg"` y `modality="image"`, `"audio"` o
`"video"`. Las plantillas en `examples/decision-{image,audio,video}.json` requieren
que aportes el archivo correspondiente. Las rutas deben quedar dentro de
`media_roots`; `media/` y las cargas temporales se excluyen de Git.

| Entrada | Procesamiento de este adaptador |
|---|---|
| Texto | Estado, pregunta, instrucciones y opciones |
| Imagen | Archivo local con extensión de imagen reconocida |
| Audio | Primeros 30 s, convertidos a mono de 16 kHz con FFmpeg |
| Vídeo | Entre 1 y 16 fotogramas muestreados, de 640 píxeles de ancho; sin pista de audio |

El límite de archivo es 32 MiB. Una petición admite texto y un archivo de una
modalidad; la interfaz comparte esa entrada entre preguntas independientes.

## API y presupuesto de tokens

| Método | Ruta E2B | Uso |
|---|---|---|
| GET | `/` | Interfaz de preguntas |
| GET | `/health` | Estado e identidad del motor |
| POST | `/count-tokens` | Mismo cuerpo que `/predict`; cuenta sin inferencia |
| POST | `/predict` | Comprueba el presupuesto y calcula probabilidades |
| POST | `/uploads?filename=...` | Sube bytes del archivo y devuelve una ruta/copia propia |
| DELETE | `/uploads/{upload_id}` | Retira la copia y libera su preparación reutilizable |

`/count-tokens` incluye plantilla de conversación, instrucciones, opciones y
contenido multimodal preparado. Devuelve `input_tokens`, `reserved_output_tokens`,
`total_tokens`, `available_tokens`, `overflow_tokens` y `fits_context`.

E2B reserva un token de respuesta: **4095 de entrada + 1 de salida = 4096**.
`/predict` repite la comprobación; rechaza un exceso con HTTP 422 y el código
`context_limit_exceeded` antes de inferir. Si no puede verificar el conteo, devuelve
un error. El presupuesto se aplica por pregunta, aunque la suma del grupo sea mayor.

La reutilización no resta tokens del presupuesto. `reuse_media=true` reutiliza la
preparación en una caché limitada a 64 MiB y 300 s; valida contenido y permisos de
ruta antes de usarla. `cache_prompt=true` permite reutilizar el prefijo del motor.
No se añaden respuestas anteriores al prompt. Para medir sin reutilización, establece
ambos campos en `false`.

Los servicios escuchan en `127.0.0.1`. Están destinados a uso local de confianza;
no tienen autenticación de usuarios ni aislamiento de tenants para exposición pública.
Las consultas pueden acceder a archivos dentro de `media_roots`. El token administrativo
se genera localmente en `integration/` y solo se usa para la parada del servicio.

## E4B y evaluación en español

E4B usa su propio modelo y proyector, con puertos independientes:

```powershell
python scripts/download_gemma_e4b_qat.py
python -m jev_qat start
python -m jev_qat predict --input examples/decision-text.json
python -m jev_qat stop
```

Su API está en http://127.0.0.1:18165/docs. La interfaz de varias preguntas y el
conteo preventivo descritos arriba corresponden a **E2B**. Ambos modelos pueden
permanecer cargados si el equipo dispone de memoria suficiente.

Se conserva el código de evaluación, calibración y candidatos del clasificador
español; los artefactos entrenados y los corpus se generan por separado. Consulta
[docs/experiments.md](docs/experiments.md) para reproducir la evaluación inicial y
entender las dependencias de los experimentos posteriores. No hay una calibración
validada para las consultas genéricas ni para E2B.

## Pruebas y mediciones

Las pruebas unitarias usan motores simulados y datos temporales; no requieren
modelos, GPU, FFmpeg ni acceso de red:

```powershell
python -m pip install -e ".[test,evaluation]"
python -m pytest tests -q
node --test scripts/test_e2b_questions.cjs
```

GitHub Actions ejecuta estas pruebas y comprueba que el wheel incluya las etiquetas
y los archivos HTML, CSS y JavaScript. La inferencia real se verifica por separado.

El [benchmark con 5, 10 y 20 preguntas](benchmarks/scaling.md) incluye tiempos, RAM,
CPU y GPU medidos en Intel Arc 140T, comprobaciones de probabilidades y controles de
4096 tokens. Son resultados de una sesión con otras aplicaciones activas; no son
pruebas de precisión semántica ni estimaciones de rendimiento en otros equipos.

## Licencia y publicación

Código bajo [MIT](LICENSE). Consulta [NOTICE.md](NOTICE.md) para atribuciones y
componentes externos. Los pesos oficiales, runtimes, corpus, medios de entrada,
perfiles generados, logs y credenciales se descargan o generan localmente y se
excluyen del repositorio.
