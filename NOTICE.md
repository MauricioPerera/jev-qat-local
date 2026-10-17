# Procedencia y componentes externos

El código de este repositorio se publica bajo [MIT](LICENSE). La licencia del
adaptador no sustituye las licencias de modelos, dependencias ni corpus externos.
No se distribuyen sus pesos, binarios, bancos de entrenamiento ni medios originales.

- El contrato de entrada y salida se inspira en
  [Jev-Omni de akhilaaa3](https://huggingface.co/akhilaaa3/Jev-Omni/blob/main/jev_omni.py).
  Este proyecto implementa su propio cálculo mediante logits de etiquetas. No carga
  la cabeza entrenada, LoRA ni pesos de Jev-Omni, y no es una distribución oficial suya.
- Modelos oficiales de Google:
  [Gemma 4 E2B QAT Q4_0](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)
  y [Gemma 4 E4B QAT Q4_0](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf).
  Los descargadores fijan revisión, tamaño y SHA-256; consulta los términos
  publicados en cada revisión del modelo antes de usarlo o redistribuirlo.
- Motor externo [llama.cpp](https://github.com/ggml-org/llama.cpp), licencia MIT;
  versión probada [b11669](https://github.com/ggml-org/llama.cpp/releases/tag/b11669).
  Se descarga por separado con sus avisos de licencia.
- [FFmpeg](https://ffmpeg.org/legal.html) y ffprobe se instalan por separado;
  sus condiciones dependen de la compilación utilizada.
- El instalador de Windows descarga [Python embebido de la PSF](https://www.python.org/downloads/release/python-3148/)
  y [FFmpeg 9.0.2 de Gyan](https://github.com/GyanD/codexffmpeg/releases/tag/9.0.2).
  Conserva sus licencias y las de los wheels en sus directorios privados. Estos
  componentes no se incluyen en el ZIP del instalador; se verifican y descargan
  desde sus distribuidores. El código MIT del adaptador no cambia sus condiciones.
- El evaluador inicial usa el corpus MASSIVE, de FitzGerald et al. / Amazon Science,
  a través de [mteb/amazon_massive_scenario](https://huggingface.co/datasets/mteb/amazon_massive_scenario).
  El código conserva atribución, revisión y hashes al preparar los datos. Los
  experimentos posteriores documentan sus corpus en los manifiestos generados.
- Los medios del benchmark se atribuyen en
  [benchmarks/ATTRIBUTION.md](benchmarks/ATTRIBUTION.md). No se incluyen los archivos.

Las bibliotecas instaladas mediante pip conservan sus propias licencias.
