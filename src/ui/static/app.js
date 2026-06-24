/* ── Marked config: code blocks framed with a language header + copy button ── */
marked.use({
  renderer: (() => {
    const r = new marked.Renderer();
    // marked@9 calls code(code, infostring); marked@12+ passes a {text, lang}
    // object. Support both so the renderer is version-proof.
    r.code = (codeArg, infostring) => {
      let text, lang;
      if (codeArg && typeof codeArg === 'object') {
        text = codeArg.text; lang = codeArg.lang;
      } else {
        text = codeArg; lang = infostring;
      }
      text = text ?? '';
      const language = lang ? String(lang).split(/\s+/)[0] : '';
      const valid = language && hljs.getLanguage(language) ? language : 'plaintext';
      const highlighted = hljs.highlight(text, { language: valid }).value;
      const label = valid === 'plaintext' ? 'text' : valid;
      return `<div class="code-block">`
        + `<div class="code-header">`
        +   `<span class="code-lang">${label}</span>`
        +   `<button class="copy-btn" type="button">${COPY_SVG}Copy</button>`
        + `</div>`
        + `<pre><code class="hljs language-${valid}">${highlighted}</code></pre>`
        + `</div>`;
    };
    return r;
  })(),
  gfm: true,
  breaks: true,
});

/* ── State ─────────────────────────────────────────────── */
let isStreaming = false;
let abortController = null;
let messageCount = 0;

/* ── DOM refs ──────────────────────────────────────────── */
const messagesEl    = document.getElementById('messages');
const scrollEl      = document.getElementById('messagesScroll');
const welcomeEl     = document.getElementById('welcomeState');
const inputEl       = document.getElementById('messageInput');
const sendBtn       = document.getElementById('sendBtn');
const statsCard     = document.getElementById('statsCard');
const uploadStatus  = document.getElementById('uploadStatus');
const dropZone      = document.getElementById('dropZone');
const fileInput     = document.getElementById('fileInput');

/* ── Icons / markup ────────────────────────────────────── */
const SEND_ICON = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>`;
const STOP_ICON = `<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2.5"/></svg>`;
const COPY_SVG  = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>`;
const BOT_AVATAR  = `<img src="/assets/bot.svg" alt="Assistant">`;
const USER_AVATAR = `<img src="/assets/user.svg" alt="You">`;
const TYPING_DOTS = `<div class="typing-dots"><span></span><span></span><span></span></div>`;

/* ── Init ──────────────────────────────────────────────── */
loadStats();
initInput();
initUpload();
initCopyDelegation();

document.getElementById('refreshBtn').addEventListener('click', loadStats);
document.getElementById('newChatBtn').addEventListener('click', newChat);

/* ── Stats ─────────────────────────────────────────────── */
async function loadStats() {
  statsCard.innerHTML = '<div class="stats-placeholder">Loading…</div>';
  try {
    const res  = await fetch('/api/stats');
    const data = await res.json();
    renderStats(data);
  } catch {
    statsCard.innerHTML = '<div class="stats-placeholder" style="color:var(--danger)">Failed to load stats</div>';
  }
}

function renderStats(data) {
  if (data.error) {
    statsCard.innerHTML = `<div class="stats-placeholder" style="color:var(--danger)">${escHtml(data.error)}</div>`;
    return;
  }
  const rows = [
    ['Collection', `<span class="stat-name">${escHtml(data.name || '—')}</span>`],
    ['Documents',  `<span class="stat-val">${data.total_documents ?? '—'}</span>`],
    ['Chunks',     `<span class="stat-val">${(data.points_count || 0).toLocaleString('en-US')}</span>`],
  ];
  if (data.total_tokens) {
    const tk = data.total_tokens;
    const s  = tk >= 1_000_000 ? `${(tk/1e6).toFixed(1)}M`
             : tk >= 1_000     ? `${(tk/1e3).toFixed(0)}K`
             : String(tk);
    rows.push(['Tokens', `<span class="stat-val">${s}</span>`]);
  }
  if (data.active_model) {
    rows.push(['Model', `<span class="stat-name" title="${escHtml(data.active_model)}">${escHtml(data.active_model)}</span>`]);
  }
  const rowsHtml = rows.map(([k, v]) =>
    `<div class="stat-row"><span class="stat-key">${k}</span>${v}</div>`
  ).join('');
  const updated = data.last_updated
    ? `<div class="stat-updated">Updated: ${escHtml(data.last_updated)}</div>` : '';
  statsCard.innerHTML = rowsHtml + updated;
}

/* ── New chat ──────────────────────────────────────────── */
async function newChat() {
  if (isStreaming) { abortController?.abort(); }
  try { await fetch('/api/reset', { method: 'POST' }); } catch {}
  messagesEl.innerHTML = '';
  messageCount = 0;
  welcomeEl.style.display = '';
  inputEl.value = '';
  inputEl.style.height = 'auto';
  sendBtn.disabled = true;
  inputEl.focus();
}

/* ── Input setup ───────────────────────────────────────── */
function initInput() {
  inputEl.addEventListener('input', onInputChange);
  inputEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (!sendBtn.disabled && !isStreaming) sendMessage();
    }
  });
  sendBtn.addEventListener('click', () => {
    if (isStreaming) stopStream();
    else if (!sendBtn.disabled) sendMessage();
  });
  document.querySelectorAll('.suggestion').forEach(btn =>
    btn.addEventListener('click', () => {
      inputEl.value = btn.dataset.text;
      onInputChange();
      sendMessage();
    })
  );
}

function onInputChange() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + 'px';
  sendBtn.disabled = !inputEl.value.trim() || isStreaming;
}

/* ── Send message ──────────────────────────────────────── */
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isStreaming) return;

  if (messageCount === 0) welcomeEl.style.display = 'none';
  messageCount++;

  appendUserMessage(text);
  inputEl.value = '';
  inputEl.style.height = 'auto';
  sendBtn.disabled = true;

  setStreaming(true);
  abortController = new AbortController();

  const { contentEl } = appendAssistantMessage();
  let fullText = '';
  let firstToken = true;

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
      signal: abortController.signal,
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const reader  = response.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';

    outer: while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();   // keep incomplete last line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6);
        if (payload === '[DONE]') break outer;
        try {
          const { token, error } = JSON.parse(payload);
          if (error) {
            contentEl.innerHTML = `<span style="color:var(--danger)">⚠️ ${escHtml(error)}</span>`;
            fullText = '';
            break outer;
          }
          if (token) {
            fullText += token;
            firstToken = false;
            contentEl.innerHTML = marked.parse(fullText);
            scrollBottom();
          }
        } catch { /* ignore malformed SSE events */ }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      contentEl.innerHTML = `<span style="color:var(--danger)">⚠️ Error: ${escHtml(err.message)}</span>`;
    } else if (firstToken) {
      contentEl.innerHTML = '<span style="color:var(--muted)">Stopped.</span>';
    }
  } finally {
    if (fullText) contentEl.innerHTML = marked.parse(fullText);
    setStreaming(false);
    scrollBottom();
    inputEl.focus();
  }
}

function stopStream() {
  abortController?.abort();
}

function setStreaming(active) {
  isStreaming = active;
  if (active) {
    sendBtn.innerHTML = STOP_ICON;
    sendBtn.classList.add('stop');
    sendBtn.disabled = false;
    sendBtn.title = 'Stop';
  } else {
    sendBtn.innerHTML = SEND_ICON;
    sendBtn.classList.remove('stop');
    sendBtn.disabled = !inputEl.value.trim();
    sendBtn.title = 'Send';
  }
}

/* ── Message rendering ─────────────────────────────────── */
function appendUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'message-row user';
  row.innerHTML = `
    <div class="msg-avatar user">${USER_AVATAR}</div>
    <div class="msg-bubble">${escHtml(text)}</div>`;
  messagesEl.appendChild(row);
  scrollBottom();
}

function appendAssistantMessage() {
  const row = document.createElement('div');
  row.className = 'message-row assistant';
  row.innerHTML = `
    <div class="msg-avatar bot">${BOT_AVATAR}</div>
    <div class="msg-bubble"><div class="msg-content">${TYPING_DOTS}</div></div>`;
  messagesEl.appendChild(row);
  scrollBottom();
  return { row, contentEl: row.querySelector('.msg-content') };
}

/* Copy buttons inside code blocks — event delegation survives stream re-renders */
function initCopyDelegation() {
  messagesEl.addEventListener('click', e => {
    const btn = e.target.closest('.copy-btn');
    if (!btn) return;
    const code = btn.closest('.code-block')?.querySelector('code');
    if (!code) return;
    navigator.clipboard.writeText(code.textContent).then(() => {
      const prev = btn.innerHTML;
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(() => { btn.innerHTML = prev; btn.classList.remove('copied'); }, 2000);
    });
  });
}

function scrollBottom() {
  scrollEl.scrollTop = scrollEl.scrollHeight;
}

/* ── Upload ────────────────────────────────────────────── */
function initUpload() {
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) uploadFiles(fileInput.files);
  });
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const pdfs = [...e.dataTransfer.files].filter(f => f.name.toLowerCase().endsWith('.pdf'));
    if (pdfs.length) uploadFiles(pdfs);
  });
}

async function uploadFiles(files) {
  uploadStatus.innerHTML = '<span style="color:var(--muted)">Indexing…</span>';
  const form = new FormData();
  for (const f of files) form.append('files', f);
  try {
    const res  = await fetch('/api/upload', { method: 'POST', body: form });
    const data = await res.json();
    const lines = data.results.map(r =>
      r.status === 'ok'
        ? `<div class="ok">✓ ${escHtml(r.file)} — ${r.chunks} chunks</div>`
        : `<div class="err">✗ ${escHtml(r.file)} — ${escHtml(r.message)}</div>`
    );
    lines.push(`<div class="total">Total: ${data.total} chunks indexed</div>`);
    uploadStatus.innerHTML = lines.join('');
    loadStats();
  } catch (e) {
    uploadStatus.innerHTML = `<span class="err">Error: ${escHtml(e.message)}</span>`;
  }
}

/* ── Helpers ───────────────────────────────────────────── */
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
