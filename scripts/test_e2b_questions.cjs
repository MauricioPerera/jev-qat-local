const test = require('node:test');
const assert = require('node:assert/strict');
const logic = require('../jev_qat/web/e2b.js');
const shared = {state: 'Añade pan. No es urgente.', video_frames: 4};
const category = {question: '¿Qué categoría?', options: 'Listas\nNoticias\nGeneral'};
const urgency = {question: '¿Es urgente?', options: 'Sí\nNo'};
const file = {name: 'same-file.png'};
const receipt = (question, budget = {fits_context: true}) => ({key: logic.key(shared, question), file, budget});

test('each request contains only its own question and options, with the shared input', () => {
  const first = logic.request(shared, category), second = logic.request(shared, urgency);
  assert.equal(first.state, second.state);
  assert.deepEqual(first.options, ['Listas', 'Noticias', 'General']);
  assert.deepEqual(second.options, ['Sí', 'No']);
  assert.equal(first.question, category.question);
  assert.equal(second.question, urgency.question);
  assert.equal(first.cache_prompt, false);
  assert.equal(second.cache_prompt, false);
  assert.ok(!JSON.stringify(second).includes('Listas'));
});
test('editing one question invalidates only its receipt', () => {
  const first = receipt(category), second = receipt(urgency);
  assert.equal(logic.ready(first, shared, {...category, question: '¿Otra categoría?'}, file), false);
  assert.equal(logic.ready(second, shared, urgency, file), true);
  assert.equal(logic.ready(first, shared, {...category, options: 'X\nY'}, file), false);
});
test('shared text or frame changes invalidate every question', () => {
  for (const question of [category, urgency]) {
    assert.equal(logic.ready(receipt(question), {...shared, state: 'Nuevo texto'}, question, file), false);
    assert.equal(logic.ready(receipt(question), {...shared, video_frames: 16}, question, file), false);
  }
});
test('a different file object invalidates the receipt even with the same filename', () => {
  assert.equal(logic.ready(receipt(category), shared, category, {name: file.name}), false);
});
test('oversized or unchecked questions cannot run, while valid questions remain eligible', () => {
  const blocked = receipt(category, {fits_context: false});
  assert.equal(logic.ready(blocked, shared, category, file), false);
  assert.equal(logic.ready(null, shared, category, file), false);
  assert.equal(logic.ready({budget: null}, shared, category, file), false);
  assert.equal(logic.ready(receipt(urgency), shared, urgency, file), true);
  assert.equal(logic.ready(receipt(category, {fits_context: 'true'}), shared, category, file), false);
});
test('all questions refer to the same upload, but have independent option arrays', () => {
  const context = {...shared, media: '/local/unique-upload.png', modality: 'image'};
  const first = logic.request(context, category), second = logic.request(context, urgency);
  assert.equal(first.media, second.media);
  assert.equal(first.modality, 'image');
  assert.equal(first.cache_prompt, true);
  assert.equal(second.cache_prompt, true);
  assert.equal(first.reuse_media, true);
  first.options.push('Mutación');
  assert.deepEqual(second.options, ['Sí', 'No']);
  assert.deepEqual(logic.request(context, category).options, ['Listas', 'Noticias', 'General']);
});
for (const question of [
  {question: '', options: 'Sí\nNo'},
  {question: '¿Qué?', options: 'Solo'},
  {question: '¿Qué?', options: 'Sí\n Sí '},
  {question: '¿Qué?', options: Array.from({length: 257}, (_, i) => String(i)).join('\n')},
]) test(`invalid question fails its own validation: ${question.options.length} chars`, () => {
  assert.throws(() => logic.request(shared, question));
  assert.doesNotThrow(() => logic.request(shared, urgency));
});
test('equal option names across different questions are allowed', () => {
  assert.doesNotThrow(() => logic.request(shared, urgency));
  assert.doesNotThrow(() => logic.request(shared, {...urgency, question: '¿Está completo?'}));
});
test('256 options are allowed for each question, not summed across questions', () => {
  const question = {question: '¿Cuál?', options: Array.from({length: 256}, (_, i) => String(i)).join('\n')};
  assert.equal(logic.request(shared, question).options.length, 256);
  assert.equal(logic.request(shared, {...question, question: '¿Y esta?'}).options.length, 256);
});
