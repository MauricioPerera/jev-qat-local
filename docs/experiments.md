# Evaluación y experimentos

El repositorio contiene el código de evaluación y de los candidatos históricos
del clasificador de mensajes en español. Los corpus, bancos joblib, perfiles de
clasificación y calibraciones generadas se conservan fuera de Git. La API genérica
de decisiones y la interfaz E2B funcionan sin esos artefactos.

## Evaluación inicial reproducible de E4B

Desde la raíz del checkout, con E4B instalado:

```powershell
python -m pip install -e ".[evaluation]"
python -m jev_qat start
python -m jev_qat.evaluate prepare
python -m jev_qat.evaluate run
python -m jev_qat.evaluate report
```

`prepare` descarga una revisión fija de MASSIVE en español, comprueba sus hashes,
elimina duplicados/conflictos y separa calibración, validación y prueba. Las etiquetas
no se envían al modelo. `run` ajusta temperatura con calibración, toma la decisión
con validación y después evalúa la prueba. Los manifiestos y resultados se escriben
en `evaluation/`, y los perfiles aprobados en `calibration/`.

Métricas: exactitud, precisión/recall/F1 por clase y macro, NLL, Brier y ECE. La
temperatura conserva la clase ganadora: calibra los puntajes, no corrige errores
de clasificación. La evaluación sobre corpus públicos no demuestra ausencia de
contaminación del preentrenamiento del modelo.

## Clasificador español y versiones posteriores

El código de `improve_spanish`, `focus_*`, `general_*`, `public_eval_*` y
`stable_rescue_*` conserva las implementaciones de los experimentos v2–v9 y sus
reglas de promoción. Son herramientas de investigación dependientes de los
artefactos de etapas anteriores. Los scripts `smoke_spanish_*` y
`verify_spanish_*` también requieren esos artefactos; no forman parte del arranque
rápido ni de la suite de pruebas unitarias.

La instalación local original mantiene v2 como versión activa. Una copia nueva de
este repositorio no contiene ningún perfil entrenado/calibrado ni activa candidatos
por defecto. Las peticiones `examples/decision-massive-*.json` son plantillas para
cuando se hayan reconstruido y validado sus perfiles correspondientes.

Las calibraciones E4B no son válidas para E2B; E2B las rechaza. Sus probabilidades
genéricas se devuelven sin calibrar. No deben reutilizarse pruebas observadas durante
el desarrollo como si fueran nuevas pruebas independientes. Los criterios de
significación, preservación de regresiones y calidad de la categoría general se
comprueban por separado antes de promover un candidato.

## Benchmarks multimodales

Consulta [el informe de escalado](../benchmarks/scaling.md) para resultados,
limitaciones y comandos de una nueva medición de 5, 10 y 20 preguntas. El código
restante de `scripts/` conserva mediciones históricas; varios scripts necesitan
planes o fuentes producidos por mediciones anteriores. La CI ejecuta únicamente
`tests/` y `scripts/test_e2b_questions.cjs`: no arranca motores ni descarga pesos.
