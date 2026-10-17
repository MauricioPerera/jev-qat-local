"use strict";
// The pure request/receipt rules are also exercised by Node's built-in tests.
const E2BQuestions = {
  options(text) { return text.split(/\r?\n/).map(value => value.trim()).filter(Boolean); },
  key(shared, question) { return JSON.stringify([shared.state, shared.video_frames, question.question, question.options]); },
  request(shared, question) {
    const prompt = question.question.trim(), options = E2BQuestions.options(question.options);
    if (!prompt) throw new Error("Escribe la pregunta que quieres resolver.");
    if (options.length < 2 || options.length > 256) throw new Error("Introduce entre 2 y 256 opciones, una por línea.");
    if (new Set(options).size !== options.length) throw new Error("Las opciones de esta pregunta deben ser distintas.");
    const payload = {state: shared.state, question: prompt, options, modality: shared.modality || "text", video_frames: shared.video_frames, cache_prompt: Boolean(shared.media), reuse_media: true};
    if (shared.media) payload.media = shared.media;
    return payload;
  },
  ready(receipt, shared, question, file) {
    return receipt?.budget?.fits_context === true && receipt.key === E2BQuestions.key(shared, question) && receipt.file === file;
  }
};
if (typeof module !== "undefined" && module.exports) module.exports = E2BQuestions;
if (typeof document !== "undefined") (() => {
  const $ = id => document.getElementById(id);
  const fileInput = $("media-file");
  const extensions = {image: ["jpg", "jpeg", "png", "webp"], audio: ["wav", "mp3", "ogg", "flac", "m4a", "aac", "opus"], video: ["mp4", "webm", "mov", "mkv", "avi", "m4v"]};
  const number = new Intl.NumberFormat("es-MX"), percent = new Intl.NumberFormat("es-MX", {style: "percent", maximumFractionDigits: 1});
  let busy = false, uploaded = null, previewURL = null, nextId = 1;
  const questions = [];
  const currentFile = () => fileInput.files[0] || null;
  const shared = () => ({state: $("state").value, video_frames: Number($("video-frames").value)});
  const values = row => ({question: row.prompt.value, options: row.options.value});
  const ready = row => E2BQuestions.ready(row.verified, shared(), values(row), currentFile());
  const status = (row, text, tone = "neutral") => { row.status.textContent = text; row.status.dataset.tone = tone; };
  function globalError(text) { $("error").textContent = text; $("error").hidden = !text; }
  function rowError(row, text) { row.error.textContent = text; row.error.hidden = !text; }
  function summary(text) {
    if (text) { $("batch-summary").textContent = text; return; }
    const enabled = questions.filter(ready).length;
    const blocked = questions.filter(row => row.verified?.budget.fits_context === false).length;
    const pending = questions.length - enabled - blocked;
    $("batch-summary").textContent = `${enabled} ${enabled === 1 ? "habilitada" : "habilitadas"} · ${blocked} ${blocked === 1 ? "bloqueada" : "bloqueadas"} · ${pending} ${pending === 1 ? "pendiente" : "pendientes"} de comprobar`;
  }
  function sync() {
    $("fields").disabled = busy; $("add-question").disabled = busy; $("check-tokens").disabled = busy;
    const enabled = questions.filter(ready).length;
    $("infer").disabled = busy || enabled === 0; $("infer").textContent = `Inferir habilitadas (${enabled})`;
    questions.forEach((row, index) => {
      row.fieldset.disabled = busy; row.count.disabled = busy; row.infer.disabled = busy || !ready(row);
      row.remove.disabled = busy || questions.length === 1;
      row.title.textContent = `Pregunta ${index + 1}`;
      row.count.setAttribute("aria-label", `Comprobar pregunta ${index + 1}`);
      row.infer.setAttribute("aria-label", `Inferir pregunta ${index + 1}`);
      row.remove.setAttribute("aria-label", `Eliminar pregunta ${index + 1}`);
      row.meter.setAttribute("aria-label", `Uso del contexto de la pregunta ${index + 1}`);
    });
    if (!busy) summary();
  }
  function invalidate(row) {
    row.verified = null; row.result.hidden = true; rowError(row, "");
    row.available.textContent = "—"; row.consumption.textContent = "Entrada: — · Respuesta: 1 · Límite: 4096";
    row.fill.style.width = "0%"; row.meter.setAttribute("aria-valuenow", "0"); row.meter.dataset.overflow = "false";
    row.help.textContent = "Incluye la entrada compartida, esta pregunta y sus opciones.";
    row.optionCount.textContent = `${E2BQuestions.options(row.options.value).length} opciones`;
    status(row, "Pendiente de comprobar");
  }
  function invalidateAll() { questions.forEach(invalidate); globalError(""); sync(); }
  function addQuestion(question = "", options = "") {
    const card = $("question-template").content.firstElementChild.cloneNode(true);
    const find = field => card.querySelector(`[data-field="${field}"]`);
    const row = {id: nextId++, card, fieldset: card.querySelector("fieldset"), title: find("title"), prompt: find("question"), options: find("options"), optionCount: find("option-count"), status: find("status"), available: find("available"), consumption: find("consumption"), meter: find("meter"), fill: find("meter-fill"), help: find("help"), result: find("result"), prediction: find("prediction"), confidence: find("confidence"), probabilities: find("probabilities"), error: card.querySelector(".question-error"), count: card.querySelector('[data-action="count"]'), infer: card.querySelector('[data-action="infer"]'), remove: card.querySelector('[data-action="remove"]'), verified: null};
    card.dataset.questionId = String(row.id);
    row.prompt.id = `question-${row.id}`; row.options.id = `options-${row.id}`;
    card.querySelector('[data-label="question"]').htmlFor = row.prompt.id;
    card.querySelector('[data-label="options"]').htmlFor = row.options.id;
    row.title.id = `question-title-${row.id}`; card.setAttribute("aria-labelledby", row.title.id);
    row.prompt.value = question; row.options.value = options;
    [row.prompt, row.options].forEach(field => field.addEventListener("input", () => { invalidate(row); sync(); }));
    row.count.addEventListener("click", () => countQuestions([row]));
    row.infer.addEventListener("click", () => inferQuestions([row]));
    row.remove.addEventListener("click", () => { if (busy || questions.length === 1) return; questions.splice(questions.indexOf(row), 1); card.remove(); sync(); });
    questions.push(row); $("questions").append(card); invalidate(row); sync(); return row;
  }
  function modality(file) {
    const extension = file.name.split(".").pop().toLowerCase();
    return Object.keys(extensions).find(key => extensions[key].includes(extension));
  }
  function validateFile(file) {
    if (!modality(file)) throw new Error("Elige una imagen JPG, PNG o WebP, un archivo de audio o un vídeo compatible.");
    if (!file.size) throw new Error("El archivo está vacío.");
    if (file.size > 32 * 1024 * 1024) throw new Error("El archivo supera el límite de 32 MiB. Selecciona uno más pequeño.");
  }
  async function jsonRequest(url, body) {
    const response = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) {
      const detail = data.detail;
      const message = Array.isArray(detail) ? detail.map(item => item.msg).join(". ") : detail && typeof detail === "object" ? detail.message : detail;
      const error = new Error(message || "No se pudo completar la solicitud."); error.detail = detail; throw error;
    }
    return data;
  }
  async function releaseUpload() {
    const old = uploaded; uploaded = null;
    if (old) { const response = await fetch(`/uploads/${old.receipt.upload_id}`, {method: "DELETE"}); if (!response.ok) throw new Error("No se pudo borrar la copia local del archivo."); }
  }
  async function prepareUpload(file) {
    if (uploaded?.file === file) return uploaded.receipt;
    await releaseUpload(); summary("Cargando el archivo compartido…");
    const response = await fetch(`/uploads?filename=${encodeURIComponent(file.name)}`, {method: "POST", headers: {"Content-Type": "application/octet-stream"}, body: file});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "No se pudo cargar el archivo.");
    uploaded = {file, receipt: data}; return data;
  }
  function renderBudget(row, budget) {
    row.available.textContent = number.format(budget.available_tokens);
    row.consumption.textContent = `Entrada: ${number.format(budget.input_tokens)} · Respuesta: ${budget.reserved_output_tokens} · Límite: ${number.format(budget.context_limit_tokens)}`;
    const total = Math.min(budget.total_tokens, budget.context_limit_tokens);
    row.fill.style.width = `${100 * total / budget.context_limit_tokens}%`;
    row.meter.setAttribute("aria-valuemax", String(budget.context_limit_tokens)); row.meter.setAttribute("aria-valuenow", String(total)); row.meter.dataset.overflow = String(!budget.fits_context);
    row.help.textContent = budget.fits_context ? "Cabe en el contexto, reservando un token para su respuesta." : `Supera el límite por ${number.format(budget.overflow_tokens)} tokens. Esta pregunta no se enviará a inferir.`;
    status(row, budget.fits_context ? "Lista para inferir" : "Supera el límite · bloqueada", budget.fits_context ? "good" : "bad");
  }
  async function countQuestions(targets) {
    if (busy) return;
    globalError(""); const context = shared(), file = currentFile(), jobs = [];
    targets.forEach(row => {
      invalidate(row);
      try { const question = values(row); E2BQuestions.request(context, question); jobs.push({row, question, key: E2BQuestions.key(context, question)}); }
      catch (error) { status(row, "Revisa esta pregunta", "bad"); rowError(row, error.message); }
    });
    try { if (file) validateFile(file); } catch (error) { globalError(error.message); sync(); return; }
    if (!jobs.length) { sync(); return; }
    busy = true; sync();
    try {
      if (file) { const receipt = await prepareUpload(file); context.media = receipt.media; context.modality = receipt.modality; }
      for (const [index, job] of jobs.entries()) {
        const {row, key} = job; summary(`Comprobando ${index + 1} de ${jobs.length} preguntas…`); status(row, "Contando texto y medios…", "busy");
        try {
          const request = E2BQuestions.request(context, job.question), budget = await jsonRequest("/count-tokens", request);
          if (!questions.includes(row) || key !== E2BQuestions.key(shared(), values(row)) || file !== currentFile()) { invalidate(row); continue; }
          row.verified = {key, file, request, budget}; renderBudget(row, budget);
        } catch (error) { row.verified = null; status(row, "No se pudo comprobar", "bad"); rowError(row, error.message); }
      }
    } catch (error) { globalError(error.message); }
    finally { busy = false; sync(); }
  }
  function renderResult(row, result) {
    row.prediction.textContent = result.prediction; row.confidence.textContent = `${percent.format(result.confidence)} de probabilidad asignada`;
    row.probabilities.replaceChildren();
    for (const [option, probability] of Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])) {
      const line = document.createElement("div"); line.className = "probability-row"; line.dataset.best = String(option === result.prediction);
      const label = document.createElement("div"); label.className = "probability-label";
      const text = document.createElement("span"), score = document.createElement("span"); text.textContent = option; score.textContent = percent.format(probability); label.append(text, score);
      const bar = document.createElement("div"), fill = document.createElement("span"); bar.className = "probability-bar"; fill.style.width = `${Math.min(100, Math.max(0, probability * 100))}%`; bar.append(fill); line.append(label, bar); row.probabilities.append(line);
    }
    row.result.hidden = false;
  }
  async function inferQuestions(targets) {
    if (busy) return;
    const eligible = targets.filter(ready);
    if (!eligible.length) { globalError("Comprueba los tokens de las preguntas que quieres inferir. Deben caber en el contexto."); sync(); return; }
    busy = true; sync(); globalError("");
    try {
      for (const [index, row] of eligible.entries()) {
        if (!ready(row)) { invalidate(row); continue; }
        const checked = row.verified; row.result.hidden = true; rowError(row, ""); status(row, "E2B está evaluando…", "busy"); summary(`Resolviendo ${index + 1} de ${eligible.length} preguntas habilitadas…`);
        try {
          const result = await jsonRequest("/predict", checked.request);
          if (ready(row) && checked === row.verified) { renderResult(row, result); renderBudget(row, checked.budget); }
          else invalidate(row);
        } catch (error) {
          row.verified = null;
          if (error.detail?.code === "context_limit_exceeded") {
            const budget = {...error.detail, available_tokens: 0, overflow_tokens: -error.detail.remaining_tokens, fits_context: false};
            row.verified = {...checked, budget}; renderBudget(row, budget);
          } else status(row, "Vuelve a comprobar esta pregunta", "bad");
          rowError(row, error.message);
        }
      }
    } finally { busy = false; sync(); }
  }
  function showFile() {
    invalidateAll(); if (previewURL) URL.revokeObjectURL(previewURL);
    previewURL = null; $("media-preview").replaceChildren();
    const file = currentFile(), kind = file && modality(file);
    $("file-details").hidden = !file; $("media-preview").hidden = !kind; $("video-settings").hidden = kind !== "video"; $("media-note").hidden = !file;
    if (!file) return;
    $("file-name").textContent = file.name;
    $("file-info").textContent = `${({image: "Imagen", audio: "Audio", video: "Vídeo"})[kind] || "Archivo"} · ${(file.size / 1024 / 1024).toFixed(2)} MiB`;
    $("media-note").textContent = kind === "audio" ? "Se prepara hasta los primeros 30 segundos una sola vez para reutilizarlos entre preguntas. Cada pregunta conserva su conteo y probabilidades." : kind === "video" ? "Se extraen una sola vez los fotogramas seleccionados para reutilizarlos entre preguntas. No se procesa la pista de audio del vídeo." : "La preparación de esta imagen se reutiliza. Cada pregunta conserva su propio contexto, opciones y probabilidades.";
    try { validateFile(file); } catch (error) { globalError(error.message); return; }
    previewURL = URL.createObjectURL(file);
    const element = document.createElement(({image: "img", audio: "audio", video: "video"})[kind]); element.src = previewURL;
    if (kind === "image") element.alt = `Archivo seleccionado: ${file.name}`; else { element.controls = true; element.preload = "metadata"; if (kind === "video") element.muted = true; }
    $("media-preview").append(element);
  }
  [$("state"), $("video-frames")].forEach(field => field.addEventListener("input", invalidateAll));
  fileInput.addEventListener("change", showFile);
  $("remove-file").addEventListener("click", async () => { fileInput.value = ""; showFile(); try { await releaseUpload(); } catch (error) { globalError(error.message); } });
  $("add-question").addEventListener("click", () => { if (!busy) addQuestion().prompt.focus(); });
  $("example").addEventListener("click", async () => {
    fileInput.value = ""; showFile(); questions.splice(0); $("questions").replaceChildren();
    $("state").value = "Añade pan a mi lista de compras para hoy. No es urgente.";
    addQuestion("¿Qué tipo de solicitud es?", "Listas\nNoticias\nGeneral"); addQuestion("¿El mensaje indica que es urgente?", "Sí\nNo");
    try { await releaseUpload(); } catch (error) { globalError(error.message); }
  });
  $("dropzone").addEventListener("dragover", event => { event.preventDefault(); if (!busy) $("dropzone").classList.add("dragging"); });
  $("dropzone").addEventListener("dragleave", () => $("dropzone").classList.remove("dragging"));
  $("dropzone").addEventListener("drop", event => { event.preventDefault(); $("dropzone").classList.remove("dragging"); if (busy) return; if (event.dataTransfer.files.length !== 1) { globalError("Selecciona un solo archivo compartido."); return; } fileInput.files = event.dataTransfer.files; showFile(); });
  $("check-tokens").addEventListener("click", () => countQuestions([...questions])); $("infer").addEventListener("click", () => inferQuestions([...questions]));
  $("workbench").addEventListener("submit", event => event.preventDefault());
  fetch("/health").then(response => { if (!response.ok) throw new Error(); return response.json(); }).then(data => { $("connection").textContent = data.status === "ready" ? "E2B listo" : "No disponible"; $("connection").dataset.ready = String(data.status === "ready"); }).catch(() => { $("connection").textContent = "Sin conexión"; });
  addQuestion("¿Qué opción describe mejor la entrada?", "");
})();
