// Run the generated preview's playback logic without a browser or network.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../..');
const previewPath = path.resolve(root, process.argv[2] || 'outputs/data_check/skeleton_preview.html');
const html = fs.readFileSync(previewPath, 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
const drawing = new Proxy({}, { get: () => () => {}, set: () => true });
function element() {
  return { textContent: '', value: '1', dataset: {}, children: [],
    classList: { toggle() {} }, append(child) { this.children.push(child); },
    getContext() { return drawing; }, width: 720, height: 500 };
}
const context = vm.createContext({
  document: { getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); },
    createElement: element,
    querySelectorAll() { return elements.get('samples').children.filter(e => e.dataset.index !== undefined); } },
  requestAnimationFrame() {},
});
vm.runInContext(script, context);
const run = source => vm.runInContext(source, context);
assert.equal(run('DATA.clips.length'), 13);
for (let i = 0; i < 13; i++) {
  run(`selectClip(${i})`);
  assert.equal(elements.get('title').textContent, run(`DATA.clips[${i}].title`));
  run('animate(0); animate(100)');
  assert.ok(run('elapsed') > 0);
}
elements.get('seek').value = '500';
elements.get('seek').oninput();
assert.ok(Math.abs(run('elapsed / DATA.clips[selected].duration') - .5) < 1e-6);
elements.get('play').onclick();
const paused = run('elapsed');
run('animate(200); animate(400)');
assert.equal(run('elapsed'), paused);
elements.get('play').onclick();
elements.get('speed').value = '2';
run('selectClip(0); animate(0); animate(100)');
assert.ok(Math.abs(run('elapsed') - .2) < 1e-6);
run('elapsed = DATA.clips[selected].duration - .01; animate(200)');
assert.ok(run('elapsed') < .3);
console.log('PASS: 13 selections, timeline seeking, pause/resume, speed, loop boundary.');
console.log('Logic checks only; browser rendering has not been verified.');
