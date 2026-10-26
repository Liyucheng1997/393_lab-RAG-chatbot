const $ = (id) => document.getElementById(id);
const messagesEl = $('messages');
const input = $('input');
const sendBtn = $('send');
let sessionId = null;
let busy = false;

// 跟随系统深色模式
const dark = matchMedia('(prefers-color-scheme: dark)');
const applyTheme = () => (document.documentElement.dataset.theme = dark.matches ? 'dark' : 'light');
applyTheme();
dark.addEventListener('change', applyTheme);

function toast(text, type = '') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = text;
  $('toasts').appendChild(el);
  requestAnimationFrame(() => el.classList.add('show'));
  setTimeout(() => { el.classList.remove('show'); setTimeout(() => el.remove(), 300); }, 2600);
}

const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// 极简 Markdown：标题、列表、粗体、行内代码，以及引用编号 [1]
function renderMarkdown(text) {
  const inline = (s) => escapeHtml(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
    .replace(/\[(\d+)\]/g, '<span class="cite" data-ref="$1">$1</span>');
  const out = [];
  let list = null;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  for (const raw of text.split('\n')) {
    const line = raw.trimEnd();
    let m;
    if ((m = line.match(/^\s*[-*]\s+(.*)/))) {
      if (list !== 'ul') { closeList(); out.push('<ul>'); list = 'ul'; }
      out.push(`<li>${inline(m[1])}</li>`);
    } else if ((m = line.match(/^\s*\d+[.)、]\s+(.*)/))) {
      if (list !== 'ol') { closeList(); out.push('<ol>'); list = 'ol'; }
      out.push(`<li>${inline(m[1])}</li>`);
    } else if ((m = line.match(/^#{1,6}\s+(.*)/))) {
      closeList(); out.push(`<h4>${inline(m[1])}</h4>`);
    } else if (line.trim()) {
      closeList(); out.push(`<p>${inline(line)}</p>`);
    } else {
      closeList();
    }
  }
  closeList();
  return out.join('');
}

function addUserMessage(text) {
  $('empty')?.remove();
  const el = document.createElement('div');
  el.className = 'msg user';
  el.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  messagesEl.appendChild(el);
}

function addBotMessage() {
  const el = document.createElement('div');
  el.className = 'msg bot';
  el.innerHTML = `<div class="bubble">
      <div class="note" hidden></div>
      <div class="answer typing"></div>
      <div class="trace" hidden>
        <details class="sources-box"><summary></summary><div class="sources"></div></details>
        <details><summary>完整 Prompt（实际发给 Claude 的内容）</summary><div class="prompt"></div></details>
      </div>
    </div>`;
  messagesEl.appendChild(el);
  // 点击引用编号 → 展开对应片段并高亮
  el.addEventListener('click', (e) => {
    const ref = e.target.closest('.cite')?.dataset.ref;
    const src = ref && el.querySelector(`.source[data-index="${ref}"]`);
    if (!src) return;
    el.querySelector('.sources-box').open = true;
    src.scrollIntoView({ behavior: 'smooth', block: 'center' });
    src.classList.add('flash');
    setTimeout(() => src.classList.remove('flash'), 1200);
  });
  return el;
}

function renderSources(el, hits) {
  el.querySelector('.trace').hidden = false;
  el.querySelector('.sources-box summary').textContent = hits.length
    ? `检索到 ${hits.length} 个片段（按余弦相似度排序）`
    : '没有检索到足够相关的片段';
  el.querySelector('.sources').innerHTML = hits.map((h) => `
    <div class="source" data-index="${h.index}">
      <div class="source-head">
        <b>[${h.index}]</b><span>${escapeHtml(h.location)}</span>
        <span class="score-bar"><i style="width:${Math.max(0, h.score) * 100}%"></i></span>
        <span class="tabular">${h.score.toFixed(3)}</span>
      </div>
      <div class="source-text">${escapeHtml(h.text)}</div>
    </div>`).join('');
}

function renderPrompt(el, ev) {
  el.querySelector('.prompt').innerHTML = `
    <div class="prompt-label">SYSTEM</div><div class="prompt-box">${escapeHtml(ev.system)}</div>
    <div class="prompt-label">USER</div><div class="prompt-box">${escapeHtml(ev.user)}</div>`;
}

const scrollToBottom = () => { messagesEl.scrollTop = messagesEl.scrollHeight; };

async function ask(question) {
  if (busy || !question.trim()) return;
  busy = true; sendBtn.disabled = true;
  addUserMessage(question);
  const el = addBotMessage();
  const answerEl = el.querySelector('.answer');
  let answer = '';
  scrollToBottom();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, session_id: sessionId }),
    });
    if (!res.ok) throw new Error(`服务器错误 ${res.status}`);
    // 后端每行发一个 JSON 事件（NDJSON），边收边渲染
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const ev = JSON.parse(line);
        if (ev.type === 'session') sessionId = ev.id;
        else if (ev.type === 'rewrite') {
          const note = el.querySelector('.note');
          note.hidden = false;
          note.textContent = `追问已改写为：${ev.query}`;
        } else if (ev.type === 'sources') renderSources(el, ev.hits);
        else if (ev.type === 'prompt') renderPrompt(el, ev);
        else if (ev.type === 'text') { answer += ev.text; answerEl.innerHTML = renderMarkdown(answer); }
        else if (ev.type === 'error') {
          const link = ev.message.includes('API Key') ? '<button type="button" class="link-btn" data-open-settings>去设置</button>' : '';
          answerEl.insertAdjacentHTML('beforeend', `<div class="error">${escapeHtml(ev.message)}${link}</div>`);
        }
      }
      scrollToBottom();
    }
  } catch (err) {
    answerEl.insertAdjacentHTML('beforeend', `<div class="error">${escapeHtml(err.message)}</div>`);
  } finally {
    answerEl.classList.remove('typing');
    busy = false; sendBtn.disabled = false;
    input.focus();
  }
}

async function loadStats(data) {
  const s = data || await (await fetch('/api/stats')).json();
  $('version').textContent = `v${s.version} · 从零手写 RAG`;
  $('statChunks').textContent = s.chunks;
  $('statEmbedder').textContent = s.embedder.toUpperCase();
  $('statTopK').textContent = s.top_k;
  $('builtAt').textContent = `切块 ${s.chunk_size} 字 · 重叠 ${s.chunk_overlap} 字 · 建于 ${s.built_at}`;
  $('docList').innerHTML = s.sources.map((n) => `<li><span title="${escapeHtml(n)}">${escapeHtml(n)}</span></li>`).join('');
}

function resetChat() {
  sessionId = null;
  location.reload();
}

$('composer').addEventListener('submit', (e) => {
  e.preventDefault();
  const q = input.value;
  input.value = '';
  input.style.height = '';
  ask(q);
});
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $('composer').requestSubmit();
  }
});
input.addEventListener('input', () => {
  input.style.height = '';
  input.style.height = `${input.scrollHeight + 2}px`;
});
document.querySelectorAll('.suggestions .chip').forEach((b) => b.addEventListener('click', () => ask(b.textContent)));
$('newChat').addEventListener('click', resetChat);
$('showTrace').addEventListener('change', (e) => document.body.classList.toggle('hide-trace', !e.target.checked));

async function withBusy(btn, label, fn) {
  const old = btn.textContent;
  btn.disabled = true; btn.textContent = label;
  try { await fn(); } catch (err) { toast(err.message, 'error'); } finally { btn.disabled = false; btn.textContent = old; }
}

$('reindex').addEventListener('click', (e) => withBusy(e.currentTarget, '重建中…', async () => {
  const res = await fetch('/api/reindex', { method: 'POST' });
  if (!res.ok) throw new Error('重建失败');
  await loadStats(await res.json());
  sessionId = null;
  toast('索引已重建');
}));

$('uploadBtn').addEventListener('click', () => $('upload').click());
$('upload').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  withBusy($('uploadBtn'), '处理中…', async () => {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch('/api/upload', { method: 'POST', body: form });
    if (!res.ok) throw new Error((await res.json()).detail || '上传失败');
    await loadStats(await res.json());
    sessionId = null;
    toast(`已加入知识库：${file.name}`);
  });
  e.target.value = '';
});

// ---------------------------------------------------------------------------
// 设置弹窗：API Key / 接口地址 / 模型 / 思考深度。保存到服务器本机的 .env
// ---------------------------------------------------------------------------
const modal = $('settingsModal');
let settings = null;
let clearKey = false;

function renderSettingsStatus() {
  $('keyDot').className = `dot ${settings.api_key_set ? 'ok' : 'bad'}`;
  $('keyStatus').textContent = settings.api_key_set ? 'API Key 已配置' : 'API Key 未配置';
  $('modelName').textContent = settings.model;
}

function renderKeyHint() {
  const hint = $('apiKeyHint');
  if (settings.api_key_set && !clearKey) {
    hint.innerHTML = `已保存 <code>${escapeHtml(settings.api_key_hint)}</code>，留空表示不修改。<button type="button" class="link-btn" id="clearKey">清除</button>`;
    $('clearKey').onclick = () => { clearKey = true; renderKeyHint(); };
  } else if (clearKey) {
    hint.textContent = '保存后会删除已保存的 Key。';
  } else {
    hint.textContent = '在 console.anthropic.com 创建。只保存在运行本程序的电脑上（项目根目录的 .env 文件）。';
  }
}

function setEffort(value) {
  document.querySelectorAll('#effort button').forEach((b) => b.classList.toggle('active', b.dataset.value === value));
}

function openSettings() {
  clearKey = false;
  $('apiKey').value = '';
  $('apiKey').type = 'password';
  $('toggleKey').textContent = '显示';
  $('baseUrl').value = settings.base_url;
  const known = settings.models.includes(settings.model);
  $('model').innerHTML = settings.models.map((m) => `<option value="${m}">${m}</option>`).join('') + '<option value="">自定义…</option>';
  $('model').value = known ? settings.model : '';
  $('customModel').hidden = known;
  $('customModel').value = known ? '' : settings.model;
  setEffort(settings.effort);
  $('fallbacks').checked = settings.use_fallbacks;
  $('testResult').hidden = true;
  renderKeyHint();
  modal.hidden = false;
  (settings.api_key_set ? $('model') : $('apiKey')).focus();
}

const closeSettings = () => { modal.hidden = true; };

function formValues() {
  const typed = $('apiKey').value.trim();
  return {
    api_key: typed || (clearKey ? '' : null),
    base_url: $('baseUrl').value.trim(),
    model: $('model').value || $('customModel').value.trim(),
    effort: document.querySelector('#effort button.active')?.dataset.value || 'low',
    use_fallbacks: $('fallbacks').checked,
  };
}

async function postJson(url, body) {
  const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `请求失败 ${res.status}`);
  return data;
}

$('openSettings').addEventListener('click', openSettings);
messagesEl.addEventListener('click', (e) => { if (e.target.closest('[data-open-settings]')) openSettings(); });
$('closeSettings').addEventListener('click', closeSettings);
$('cancelSettings').addEventListener('click', closeSettings);
modal.addEventListener('mousedown', (e) => { if (e.target === modal) closeSettings(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !modal.hidden) closeSettings(); });
$('toggleKey').addEventListener('click', () => {
  const show = $('apiKey').type === 'password';
  $('apiKey').type = show ? 'text' : 'password';
  $('toggleKey').textContent = show ? '隐藏' : '显示';
});
$('model').addEventListener('change', () => {
  $('customModel').hidden = $('model').value !== '';
  if (!$('customModel').hidden) $('customModel').focus();
});
$('effort').addEventListener('click', (e) => { if (e.target.dataset.value) setEffort(e.target.dataset.value); });

$('testSettings').addEventListener('click', (e) => withBusy(e.currentTarget, '测试中…', async () => {
  const result = await postJson('/api/settings/test', formValues());
  const box = $('testResult');
  box.hidden = false;
  box.className = `test-result ${result.ok ? '' : 'bad'}`;
  box.textContent = result.message;
}));

$('settingsForm').addEventListener('submit', (e) => {
  e.preventDefault();
  withBusy($('saveSettings'), '保存中…', async () => {
    settings = await postJson('/api/settings', formValues());
    renderSettingsStatus();
    closeSettings();
    toast('设置已保存');
  });
});

async function loadSettings() {
  settings = await (await fetch('/api/settings')).json();
  renderSettingsStatus();
  if (!settings.api_key_set) openSettings();  // 第一次使用，直接引导填写 Key
}

loadStats().catch(() => toast('加载知识库信息失败', 'error'));
loadSettings().catch(() => toast('加载设置失败', 'error'));
input.focus();
