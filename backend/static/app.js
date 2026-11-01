/**
 * Personal LLM — Modern Linear-Inspired Workspace Client
 * 
 * Features:
 *   - LocalStorage conversation persistence
 *   - Markdown rendering with syntax-highlighted code blocks
 *   - Multi-turn conversation context using /api/v1/chat
 *   - Restrained micro-interactions & keyboard shortcuts (Cmd+K, Cmd+B)
 *   - Full parameter customization (temperature, max tokens, repetition penalty)
 */

(() => {
  'use strict';

  // ─── Constants & Storage Keys ────────────────────────────────────
  const STORAGE_KEY_CONVERSATIONS = 'personal_llm_conversations_v1';
  const STORAGE_KEY_ACTIVE_ID = 'personal_llm_active_id_v1';
  const STORAGE_KEY_SETTINGS = 'personal_llm_settings_v1';

  const DEFAULT_SETTINGS = {
    temperature: 0.7,
    max_new_tokens: 128,
    repetition_penalty: 1.15,
    apiKey: '',
  };

  // Detect if opened as file:/// or via http
  const isFileProtocol = window.location.protocol === 'file:';
  const API_BASE = isFileProtocol ? 'http://localhost:8000' : '';

  // ─── State ───────────────────────────────────────────────────────
  let conversations = [];
  let activeConversationId = null;
  let settings = { ...DEFAULT_SETTINGS };
  let isGenerating = false;

  // ─── DOM Element References ──────────────────────────────────────
  const el = {
    sidebar: document.getElementById('sidebar'),
    sidebarBackdrop: document.getElementById('sidebar-backdrop'),
    collapseSidebarBtn: document.getElementById('collapse-sidebar-btn'),
    mobileToggleBtn: document.getElementById('mobile-toggle-btn'),
    newChatBtn: document.getElementById('new-chat-btn'),
    conversationSearch: document.getElementById('conversation-search'),
    historyList: document.getElementById('history-list'),
    healthDot: document.getElementById('health-dot'),
    deviceLabel: document.getElementById('device-label'),
    openSettingsBtn: document.getElementById('open-settings-btn'),
    closeSettingsBtn: document.getElementById('close-settings-btn'),
    settingsModal: document.getElementById('settings-modal'),
    saveSettingsBtn: document.getElementById('save-settings-btn'),
    tempSlider: document.getElementById('temp-slider'),
    tempVal: document.getElementById('temp-val'),
    tokensSlider: document.getElementById('tokens-slider'),
    tokensVal: document.getElementById('tokens-val'),
    penaltySlider: document.getElementById('penalty-slider'),
    penaltyVal: document.getElementById('penalty-val'),
    apiKeyInput: document.getElementById('api-key-input'),
    activeChatTitle: document.getElementById('active-chat-title'),
    clearChatBtn: document.getElementById('clear-chat-btn'),
    exportChatBtn: document.getElementById('export-chat-btn'),
    conversationViewport: document.getElementById('conversation-viewport'),
    messagesContainer: document.getElementById('messages-container'),
    welcomeView: document.getElementById('welcome-view'),
    promptInput: document.getElementById('prompt-input'),
    charCounter: document.getElementById('char-counter'),
    tempIndicator: document.getElementById('temp-indicator'),
    sendBtn: document.getElementById('send-btn'),
  };

  // ─── Initialization ──────────────────────────────────────────────
  function init() {
    loadSettings();
    loadConversations();
    setupEventHandlers();
    checkSystemStatus();

    if (conversations.length === 0) {
      createNewConversation(false);
    } else {
      const targetId = localStorage.getItem(STORAGE_KEY_ACTIVE_ID) || conversations[0].id;
      selectConversation(targetId);
    }
  }

  // ─── Storage Operations ──────────────────────────────────────────
  function loadSettings() {
    try {
      const saved = localStorage.getItem(STORAGE_KEY_SETTINGS);
      if (saved) {
        settings = { ...DEFAULT_SETTINGS, ...JSON.parse(saved) };
      }
    } catch {
      settings = { ...DEFAULT_SETTINGS };
    }
    updateSettingsUI();
  }

  function saveSettings() {
    localStorage.setItem(STORAGE_KEY_SETTINGS, JSON.stringify(settings));
    el.tempIndicator.textContent = `temp ${settings.temperature}`;
  }

  function loadConversations() {
    try {
      const saved = localStorage.getItem(STORAGE_KEY_CONVERSATIONS);
      conversations = saved ? JSON.parse(saved) : [];
    } catch {
      conversations = [];
    }
  }

  function saveConversations() {
    localStorage.setItem(STORAGE_KEY_CONVERSATIONS, JSON.stringify(conversations));
    renderHistoryList();
  }

  // ─── Conversation Management ─────────────────────────────────────
  function createNewConversation(select = true) {
    const newConv = {
      id: 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6),
      title: 'New Conversation',
      createdAt: new Date().toISOString(),
      messages: [],
    };

    conversations.unshift(newConv);
    saveConversations();

    if (select) {
      selectConversation(newConv.id);
    }
    return newConv;
  }

  function getActiveConversation() {
    return conversations.find((c) => c.id === activeConversationId);
  }

  function selectConversation(id) {
    const target = conversations.find((c) => c.id === id);
    if (!target) return;

    activeConversationId = id;
    localStorage.setItem(STORAGE_KEY_ACTIVE_ID, id);

    el.activeChatTitle.textContent = target.title;
    renderMessages(target.messages);
    renderHistoryList();

    // Close mobile drawer on selection
    el.sidebar.classList.remove('mobile-open');
    el.sidebarBackdrop.classList.remove('active');
    el.promptInput.focus();
  }

  function deleteConversation(id, event) {
    if (event) event.stopPropagation();

    conversations = conversations.filter((c) => c.id !== id);
    saveConversations();

    if (activeConversationId === id) {
      if (conversations.length > 0) {
        selectConversation(conversations[0].id);
      } else {
        createNewConversation(true);
      }
    }
  }

  function renameConversation(id, event) {
    if (event) event.stopPropagation();

    const conv = conversations.find((c) => c.id === id);
    if (!conv) return;

    const newTitle = prompt('Rename conversation:', conv.title);
    if (newTitle && newTitle.trim()) {
      conv.title = newTitle.trim();
      saveConversations();
      if (activeConversationId === id) {
        el.activeChatTitle.textContent = conv.title;
      }
    }
  }

  function clearActiveConversation() {
    const conv = getActiveConversation();
    if (!conv || conv.messages.length === 0) return;

    if (confirm('Clear all messages in this conversation?')) {
      conv.messages = [];
      saveConversations();
      renderMessages([]);
    }
  }

  function exportActiveConversation() {
    const conv = getActiveConversation();
    if (!conv || conv.messages.length === 0) return;

    let markdown = `# ${conv.title}\n*Created: ${new Date(conv.createdAt).toLocaleString()}*\n\n---\n\n`;
    for (const msg of conv.messages) {
      const author = msg.role === 'user' ? '### User' : '### Assistant (Personal LLM)';
      markdown += `${author}\n\n${msg.content}\n\n`;
    }

    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${conv.title.toLowerCase().replace(/[^a-z0-9]+/g, '-')}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // ─── Rendering Engine ────────────────────────────────────────────
  function renderHistoryList(filter = '') {
    el.historyList.innerHTML = '';
    const query = filter.toLowerCase().trim();

    const filtered = conversations.filter((c) =>
      c.title.toLowerCase().includes(query)
    );

    if (filtered.length === 0) {
      const emptyItem = document.createElement('div');
      emptyItem.className = 'history-item';
      emptyItem.style.color = 'var(--text-muted)';
      emptyItem.style.fontSize = '12px';
      emptyItem.textContent = query ? 'No matching chats' : 'No conversations yet';
      el.historyList.appendChild(emptyItem);
      return;
    }

    for (const conv of filtered) {
      const item = document.createElement('div');
      item.className = 'history-item' + (conv.id === activeConversationId ? ' active' : '');
      item.onclick = () => selectConversation(conv.id);

      const title = document.createElement('span');
      title.className = 'history-title';
      title.textContent = conv.title;

      const actions = document.createElement('div');
      actions.className = 'history-actions';

      const editBtn = document.createElement('button');
      editBtn.className = 'item-action-btn';
      editBtn.title = 'Rename';
      editBtn.innerHTML = `
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"></path>
        </svg>
      `;
      editBtn.onclick = (e) => renameConversation(conv.id, e);

      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'item-action-btn';
      deleteBtn.title = 'Delete';
      deleteBtn.innerHTML = `
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="3 6 5 6 21 6"></polyline>
          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"></path>
        </svg>
      `;
      deleteBtn.onclick = (e) => deleteConversation(conv.id, e);

      actions.appendChild(editBtn);
      actions.appendChild(deleteBtn);

      item.appendChild(title);
      item.appendChild(actions);

      el.historyList.appendChild(item);
    }
  }

  function renderMessages(messages) {
    el.messagesContainer.innerHTML = '';

    if (!messages || messages.length === 0) {
      el.messagesContainer.appendChild(el.welcomeView);
      return;
    }

    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];
      appendMessageElement(msg.role, msg.content, msg.timestamp, i);
    }

    scrollToBottom();
  }

  function appendMessageElement(role, content, timestamp, index) {
    const row = document.createElement('div');
    row.className = `message-row ${role}`;

    const meta = document.createElement('div');
    meta.className = 'message-meta';

    const author = document.createElement('span');
    author.className = 'message-author';
    author.textContent = role === 'user' ? 'YOU' : 'PERSONAL LLM';

    const time = document.createElement('span');
    time.textContent = timestamp
      ? new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      : '';

    meta.appendChild(author);
    meta.appendChild(time);

    const body = document.createElement('div');
    body.className = 'message-body';

    if (role === 'user') {
      body.textContent = content;
    } else {
      body.innerHTML = parseMarkdown(content);
      postProcessCodeBlocks(body);
    }

    row.appendChild(meta);
    row.appendChild(body);

    if (role === 'assistant') {
      const actions = document.createElement('div');
      actions.className = 'message-actions';

      const copyBtn = document.createElement('button');
      copyBtn.className = 'action-btn';
      copyBtn.innerHTML = `
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
        <span>Copy</span>
      `;
      copyBtn.onclick = () => {
        navigator.clipboard.writeText(content);
        const span = copyBtn.querySelector('span');
        span.textContent = 'Copied';
        setTimeout(() => (span.textContent = 'Copy'), 1500);
      };

      actions.appendChild(copyBtn);
      row.appendChild(actions);
    }

    el.messagesContainer.appendChild(row);
  }

  function appendLoadingIndicator() {
    const row = document.createElement('div');
    row.id = 'active-generating-row';
    row.className = 'message-row assistant';

    const meta = document.createElement('div');
    meta.className = 'message-meta';

    const author = document.createElement('span');
    author.className = 'message-author';
    author.textContent = 'PERSONAL LLM';

    meta.appendChild(author);

    const indicator = document.createElement('div');
    indicator.className = 'generating-indicator';
    indicator.innerHTML = `
      <span class="pulse-dot"></span>
      <span>Generating response...</span>
    `;

    row.appendChild(meta);
    row.appendChild(indicator);
    el.messagesContainer.appendChild(row);
    scrollToBottom();
  }

  function removeLoadingIndicator() {
    const loadingRow = document.getElementById('active-generating-row');
    if (loadingRow) loadingRow.remove();
  }

  // ─── Markdown & Code Parser ──────────────────────────────────────
  function parseMarkdown(text) {
    if (window.marked && typeof window.marked.parse === 'function') {
      try {
        return window.marked.parse(text);
      } catch {
        // Fallback below
      }
    }
    // Basic fallback if marked.js is not loaded
    const escaped = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    return `<p>${escaped.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>')}</p>`;
  }

  function postProcessCodeBlocks(container) {
    const codeBlocks = container.querySelectorAll('pre code');
    for (const codeEl of codeBlocks) {
      if (window.hljs) {
        window.hljs.highlightElement(codeEl);
      }

      const pre = codeEl.parentElement;
      const langClass = Array.from(codeEl.classList).find((c) => c.startsWith('language-'));
      const lang = langClass ? langClass.replace('language-', '') : 'text';

      const wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';

      const header = document.createElement('div');
      header.className = 'code-header';

      const langLabel = document.createElement('span');
      langLabel.textContent = lang;

      const copyBtn = document.createElement('button');
      copyBtn.className = 'copy-code-btn';
      copyBtn.innerHTML = `
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
        <span>Copy code</span>
      `;
      copyBtn.onclick = () => {
        navigator.clipboard.writeText(codeEl.textContent);
        const span = copyBtn.querySelector('span');
        span.textContent = 'Copied!';
        setTimeout(() => (span.textContent = 'Copy code'), 1500);
      };

      header.appendChild(langLabel);
      header.appendChild(copyBtn);

      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(header);
      wrapper.appendChild(pre);
    }
  }

  function scrollToBottom() {
    el.conversationViewport.scrollTop = el.conversationViewport.scrollHeight;
  }

  // ─── Sending & API Communication ─────────────────────────────────
  async function handleSend() {
    if (isGenerating) return;

    const prompt = el.promptInput.value.trim();
    if (!prompt) return;

    const conv = getActiveConversation();
    if (!conv) return;

    // First user turn sets the conversation title
    if (conv.messages.length === 0) {
      conv.title = prompt.length > 32 ? prompt.substring(0, 32).trim() + '...' : prompt;
      el.activeChatTitle.textContent = conv.title;
    }

    // Append user message
    const timestamp = new Date().toISOString();
    conv.messages.push({ role: 'user', content: prompt, timestamp });
    saveConversations();

    // Reset composer
    el.promptInput.value = '';
    el.promptInput.style.height = 'auto';
    updateCharCounter();
    el.sendBtn.disabled = true;

    // Render user message and loading indicator
    if (conv.messages.length === 1 && el.messagesContainer.contains(el.welcomeView)) {
      el.messagesContainer.innerHTML = '';
    }
    appendMessageElement('user', prompt, timestamp, conv.messages.length - 1);
    appendLoadingIndicator();

    isGenerating = true;

    try {
      const headers = { 'Content-Type': 'application/json' };
      if (settings.apiKey) {
        headers['X-API-Key'] = settings.apiKey;
      }

      // Format payload for /api/v1/chat endpoint
      const payload = {
        messages: conv.messages.map((m) => ({ role: m.role, content: m.content })),
        max_new_tokens: Number(settings.max_new_tokens),
        temperature: Number(settings.temperature),
        repetition_penalty: Number(settings.repetition_penalty),
      };

      const response = await fetch(API_BASE + '/api/v1/chat', {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
      });

      removeLoadingIndicator();

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'HTTP ' + response.status }));
        let detailMsg = 'Generation request failed.';
        if (typeof err.detail === 'string') {
          detailMsg = err.detail;
        } else if (Array.isArray(err.detail)) {
          detailMsg = err.detail.map((e) => e.msg || JSON.stringify(e)).join(', ');
        }
        throw new Error(detailMsg);
      }

      const data = await response.json();
      const assistantContent = data.content || '[Empty response]';
      const assistantTimestamp = new Date().toISOString();

      conv.messages.push({
        role: 'assistant',
        content: assistantContent,
        timestamp: assistantTimestamp,
      });
      saveConversations();

      appendMessageElement('assistant', assistantContent, assistantTimestamp, conv.messages.length - 1);
      scrollToBottom();
    } catch (err) {
      removeLoadingIndicator();
      const errorMsg = `**Error**: ${err.message || 'Failed to connect to LLM server.'}\n\n*Check server console or /api/v1/health.*`;
      appendMessageElement('assistant', errorMsg, new Date().toISOString(), conv.messages.length);
      scrollToBottom();
    } finally {
      isGenerating = false;
      el.promptInput.focus();
    }
  }

  // ─── Input Resizing & Event Listeners ─────────────────────────────
  function setupEventHandlers() {
    // Prompt Input
    el.promptInput.addEventListener('input', () => {
      el.promptInput.style.height = 'auto';
      el.promptInput.style.height = Math.min(el.promptInput.scrollHeight, 180) + 'px';
      updateCharCounter();
      el.sendBtn.disabled = !el.promptInput.value.trim() || isGenerating;
    });

    el.promptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    });

    el.sendBtn.addEventListener('click', handleSend);

    // Sidebar Toggles
    el.collapseSidebarBtn.addEventListener('click', () => {
      el.sidebar.classList.toggle('collapsed');
    });

    el.mobileToggleBtn.addEventListener('click', () => {
      el.sidebar.classList.toggle('mobile-open');
      el.sidebarBackdrop.classList.toggle('active');
    });

    el.sidebarBackdrop.addEventListener('click', () => {
      el.sidebar.classList.remove('mobile-open');
      el.sidebarBackdrop.classList.remove('active');
    });

    // New Chat & Clear
    el.newChatBtn.addEventListener('click', () => createNewConversation(true));
    el.clearChatBtn.addEventListener('click', clearActiveConversation);
    el.exportChatBtn.addEventListener('click', exportActiveConversation);

    // Search
    el.conversationSearch.addEventListener('input', (e) => {
      renderHistoryList(e.target.value);
    });

    // Prompt Suggestions on Welcome View
    document.querySelectorAll('.suggestion-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        el.promptInput.value = chip.getAttribute('data-prompt');
        el.promptInput.dispatchEvent(new Event('input'));
        handleSend();
      });
    });

    // Settings Modal
    el.openSettingsBtn.addEventListener('click', () => el.settingsModal.classList.add('open'));
    el.closeSettingsBtn.addEventListener('click', () => el.settingsModal.classList.remove('open'));
    el.settingsModal.addEventListener('click', (e) => {
      if (e.target === el.settingsModal) el.settingsModal.classList.remove('open');
    });

    el.tempSlider.addEventListener('input', (e) => (el.tempVal.textContent = e.target.value));
    el.tokensSlider.addEventListener('input', (e) => (el.tokensVal.textContent = e.target.value));
    el.penaltySlider.addEventListener('input', (e) => (el.penaltyVal.textContent = e.target.value));

    el.saveSettingsBtn.addEventListener('click', () => {
      settings.temperature = parseFloat(el.tempSlider.value);
      settings.max_new_tokens = parseInt(el.tokensSlider.value, 10);
      settings.repetition_penalty = parseFloat(el.penaltySlider.value);
      settings.apiKey = el.apiKeyInput.value.trim();
      saveSettings();
      el.settingsModal.classList.remove('open');
    });

    // Global Keyboard Shortcuts (Cmd+K, Cmd+B, Esc)
    window.addEventListener('keydown', (e) => {
      const isCmd = e.metaKey || e.ctrlKey;
      if (isCmd && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        createNewConversation(true);
      } else if (isCmd && e.key.toLowerCase() === 'b') {
        e.preventDefault();
        el.sidebar.classList.toggle('collapsed');
      } else if (e.key === 'Escape') {
        el.settingsModal.classList.remove('open');
      }
    });
  }

  function updateCharCounter() {
    const len = el.promptInput.value.length;
    el.charCounter.textContent = `${len.toLocaleString()} / 10,000`;
    if (len > 10000) {
      el.charCounter.style.color = 'var(--error)';
    } else {
      el.charCounter.style.color = 'var(--text-muted)';
    }
  }

  function updateSettingsUI() {
    el.tempSlider.value = settings.temperature;
    el.tempVal.textContent = settings.temperature;
    el.tokensSlider.value = settings.max_new_tokens;
    el.tokensVal.textContent = settings.max_new_tokens;
    el.penaltySlider.value = settings.repetition_penalty;
    el.penaltyVal.textContent = settings.repetition_penalty;
    el.apiKeyInput.value = settings.apiKey || '';
    el.tempIndicator.textContent = `temp ${settings.temperature}`;
  }

  async function checkSystemStatus() {
    try {
      const res = await fetch(API_BASE + '/api/v1/health');
      if (res.ok) {
        const data = await res.json();
        el.healthDot.style.backgroundColor = 'var(--success)';
        el.deviceLabel.textContent = data.model_loaded ? 'Model Loaded • Ready' : 'TinyLlama • Ready';
      }
    } catch {
      el.healthDot.style.backgroundColor = 'var(--text-muted)';
      el.deviceLabel.textContent = 'Server Offline';
    }
  }

  // ─── Bootstrap ───────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', init);
})();

