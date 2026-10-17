"""Fixed Spanish questions for capacity testing, not a labeled accuracy corpus."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cases():
    original = json.loads((ROOT / "benchmarks/scaling-cases.json").read_text(encoding="utf8"))
    for case in original:
        case["source"] = str((ROOT / case["source"]).resolve())
    extra = {
        "image_alps": [
            ("¿Se aprecia agua en el paisaje?", ["sí", "no"]),
            ("¿Predomina una escena interior?", ["sí", "no"]),
            ("¿La imagen parece una captura de una aplicación informática?", ["captura de software", "fotografía de paisaje", "gráfico estadístico"]),
            ("¿Qué tipo de relieve se distingue?", ["montañoso", "llanura desértica", "fondo marino"]),
            ("¿Hay cielo visible en la parte superior?", ["sí", "no"]),
            ("¿La escena principal está bajo el agua?", ["sí", "no"]),
            ("¿Se ven zonas de nieve o cumbres de color claro?", ["sí", "no"]),
            ("¿La composición ofrece una vista amplia o un primer plano de un rostro?", ["vista amplia de paisaje", "primer plano de un rostro"]),
            ("¿Se distingue vegetación en el entorno?", ["sí", "no"]),
            ("¿La iluminación parece diurna o nocturna?", ["diurna", "nocturna"]),
            ("¿Domina una gran zona de edificios?", ["sí", "no"]),
            ("¿Qué elemento natural destaca más?", ["relieve de montaña", "arena de playa", "hielo submarino", "lava"]),
            ("¿El contenido principal es texto escrito?", ["sí", "no"]),
            ("¿Se ve el interior de una habitación?", ["sí", "no"]),
            ("¿Qué paisaje es más compatible con la fotografía?", ["paisaje alpino", "selva tropical", "desierto de dunas"]),
            ("¿El agua forma parte de una vista terrestre o de una escena submarina?", ["vista terrestre", "escena submarina"]),
            ("¿Qué tema elegirías para clasificar esta imagen?", ["naturaleza y montaña", "comida preparada", "retrato humano", "documento escaneado"]),
        ],
        "audio_30": [
            ("¿Predomina una locución o una interpretación instrumental?", ["locución hablada", "interpretación instrumental"]),
            ("¿El fragmento consiste únicamente en silencio?", ["sí", "no"]),
            ("¿La voz parece narrar o leer información?", ["narración o lectura", "canto sin palabras", "ruido de motor"]),
            ("¿Se distinguen palabras o solo sonidos sin lenguaje?", ["palabras", "solo sonidos sin lenguaje"]),
            ("¿Qué fuente sonora describe mejor el registro?", ["voz humana", "olas del mar", "motor de automóvil", "pájaros"]),
            ("¿Predomina una voz individual o una conversación entre varias personas?", ["voz individual", "conversación de varias personas"]),
            ("¿El registro principal es habla o canto musical?", ["habla", "canto musical"]),
            ("¿La pista parece ser únicamente música electrónica?", ["sí", "no"]),
            ("¿El contenido menciona la Tierra?", ["sí", "no"]),
            ("¿El tema parece científico o una receta de cocina?", ["tema científico", "receta de cocina"]),
            ("¿El idioma principal coincide con el español?", ["sí", "no"]),
            ("¿El discurso parece informativo o una orden de compra?", ["discurso informativo", "orden de compra"]),
            ("¿Se habla de características de un planeta?", ["sí", "no"]),
            ("¿Qué etiqueta asignarías al tipo de material?", ["voz hablada", "música instrumental", "ruido ambiental"]),
            ("¿La persona habla inglés como idioma principal?", ["sí", "no"]),
            ("¿Se percibe una persona hablando o un sonido mecánico repetitivo?", ["persona hablando", "sonido mecánico repetitivo"]),
            ("¿Cuál es la clasificación más adecuada para este fragmento?", ["narración", "silencio", "canción", "efecto sonoro"]),
        ],
        "video_60_16": [
            ("¿Se distinguen edificios en la escena?", ["sí", "no"]),
            ("¿Hay una masa de agua visible?", ["sí", "no"]),
            ("¿La escena se desarrolla principalmente en exteriores?", ["exterior", "interior"]),
            ("¿Qué paisaje corresponde mejor al vídeo?", ["ciudad costera", "desierto de dunas", "bosque nevado"]),
            ("¿Se observa principalmente un primer plano de una persona?", ["sí", "no"]),
            ("¿La iluminación parece diurna?", ["sí", "no"]),
            ("¿El punto de vista parece elevado respecto a los edificios?", ["sí", "no"]),
            ("¿Qué tipo de grabación describe mejor la perspectiva?", ["toma aérea", "cámara de escritorio", "microscopía", "endoscopia"]),
            ("¿Se ven estructuras construidas por personas?", ["sí", "no"]),
            ("¿Predomina una escena submarina?", ["sí", "no"]),
            ("¿Qué combinación de elementos es más compatible con las imágenes?", ["edificios y costa", "cocina y utensilios", "cama y armario"]),
            ("¿El vídeo muestra una habitación cerrada?", ["sí", "no"]),
            ("¿Qué ámbito describe mejor el lugar?", ["urbano costero", "rural sin edificios", "interior industrial"]),
            ("¿Hay zonas de tierra junto al agua?", ["sí", "no"]),
            ("¿La vista abarca un área amplia o solo un objeto pequeño?", ["área amplia", "un objeto pequeño"]),
            ("¿El contenido principal son letras de un documento?", ["sí", "no"]),
            ("¿Cuál sería una etiqueta temática adecuada?", ["vista de ciudad", "preparación de comida", "retrato en estudio", "documento"]),
        ],
    }
    for case in original:
        questions = [(q["question"], q["options"]) for q in case["questions"]] + extra[case["id"]]
        assert len(questions) == 20 and len({q for q, _ in questions}) == 20
        # Move correct/likely options between label positions; each question has its own vector.
        case["questions"] = []
        for index, (question, choices) in enumerate(questions):
            shift = index % len(choices)
            options = choices[shift:] + choices[:shift]
            case["questions"].append({"id": index, "question": question, "options": options})
    return original
