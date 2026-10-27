/* ═══════════════════════════════════════════════════════════════
   QueryFolio — Frontend Controller
   Streaming RAG with Mermaid Flowcharts & OCR Citations
   ═══════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {

    // ── DOM Element Bindings ──
    const fileInput             = document.getElementById('fileInput');
    const welcomeUploadBtn      = document.getElementById('welcomeUploadBtn');
    const welcomeDropTarget     = document.getElementById('welcomeDropTarget');
    const processingContainer   = document.getElementById('processingContainer');
    const processingCurrentStep = document.getElementById('processingCurrentStep');
    const docBlockSection       = document.getElementById('docBlockSection');
    const documentInfoCard      = document.getElementById('documentInfoCard');
    const sidebarEmptyState     = document.getElementById('sidebarEmptyState');
    const docName               = document.getElementById('docName');
    const docStats              = document.getElementById('docStats');
    const changeFileBtn         = document.getElementById('changeFileBtn');
    const questionsSection      = document.getElementById('questionsSection');
    const questionsGrid         = document.getElementById('questionsGrid');
    const chatSection           = document.getElementById('chat-section');
    const chatHistory           = document.getElementById('chatHistory');
    const chatForm              = document.getElementById('chatForm');
    const chatInput             = document.getElementById('chatInput');
    const chatWelcomeMsg        = document.getElementById('chatWelcomeMsg');
    const welcomeScreen         = document.getElementById('welcomeScreen');
    const chatDocTitle          = document.getElementById('chatDocTitle');
    const sendBtn               = document.getElementById('sendBtn');
    const clearChatBtn          = document.getElementById('clearChatBtn');
    const navChat               = document.getElementById('navChat');
    const mobileSidebarToggle   = document.getElementById('mobileSidebarToggle');
    const sidebar               = document.getElementById('sidebar');
    const attachFileBtn         = document.getElementById('attachFileBtn');

    // ── State Variables ──
    let activeDocumentId       = null;
    let isStreaming            = false;
    let isUploading            = false;
    let activeAbortController  = null;
    let conversationHistory    = [];

    // API Base URL (relative '/api' when served by FastAPI or deployed on cloud; fallback to port 8000 for separate dev live servers)
    const API_BASE = ((window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost') && window.location.port && window.location.port !== '8000')
        ? 'http://127.0.0.1:8000/api'
        : '/api';

    // ── Initialize Mermaid (Warm Beige & Burgundy Theme) ──
    if (typeof window.mermaid !== 'undefined') {
        window.mermaid.initialize({
            startOnLoad: false,
            theme: 'base',
            themeVariables: {
                background: '#F5E8E1',
                primaryColor: '#8B1E3F',
                primaryTextColor: '#FFFFFF',
                primaryBorderColor: '#6A152E',
                secondaryColor: '#E6C7C2',
                secondaryTextColor: '#3A1F1F',
                secondaryBorderColor: '#D8AEA7',
                tertiaryColor: '#FFFCF9',
                tertiaryTextColor: '#3A1F1F',
                tertiaryBorderColor: '#EADBD3',
                lineColor: '#8B1E3F',
                edgeLabelBackground: '#FEF9F4',
                nodeBorder: '#8B1E3F',
                clusterBkg: '#FFFCF9',
                clusterBorder: '#EADBD3',
                fontSize: '13px',
                fontFamily: 'Inter, system-ui, sans-serif'
            },
            flowchart: { curve: 'basis', diagramPadding: 16 },
            securityLevel: 'strict',
        });
    }

    // ── File Picker & Welcome Upload Button ──
    if (attachFileBtn && fileInput) {
        attachFileBtn.addEventListener('click', () => fileInput.click());
    }

    if (welcomeUploadBtn && fileInput) {
        welcomeUploadBtn.addEventListener('click', (e) => {
            e.preventDefault();
            fileInput.click();
        });
    }

    if (welcomeDropTarget) {
        ['dragenter', 'dragover'].forEach(evt => {
            welcomeDropTarget.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                welcomeDropTarget.classList.add('dragover');
            });
        });
        ['dragleave', 'drop'].forEach(evt => {
            welcomeDropTarget.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                welcomeDropTarget.classList.remove('dragover');
            });
        });
        welcomeDropTarget.addEventListener('drop', (e) => {
            e.preventDefault();
            const files = e.dataTransfer.files;
            if (files && files.length > 0) handleUpload(files[0]);
        });
    }

    // ── Pipeline Step Progression (7 Required Stages) ──
    const STEPS = [
        { id: 'step-upload',    label: 'Uploading…' },
        { id: 'step-read',      label: 'Reading document…' },
        { id: 'step-ocr',       label: 'OCR / vision processing…' },
        { id: 'step-chunk',     label: 'Chunking…' },
        { id: 'step-embed',     label: 'Embedding…' },
        { id: 'step-index',     label: 'Indexing…' },
        { id: 'step-questions', label: 'Generating questions…' },
    ];

    function resetPipelineSteps() {
        STEPS.forEach(s => {
            const el = document.getElementById(s.id);
            if (el) el.className = 'pipe-step';
        });
    }

    function setStepActive(stepId, label) {
        const el = document.getElementById(stepId);
        if (el) el.className = 'pipe-step active';
        if (processingCurrentStep) processingCurrentStep.textContent = label;
    }

    function setStepDone(stepId) {
        const el = document.getElementById(stepId);
        if (el) el.className = 'pipe-step done';
    }

    function markAllPipelineDone() {
        STEPS.forEach(s => setStepDone(s.id));
        if (processingCurrentStep) processingCurrentStep.textContent = 'Document ready ✓';
    }

    function animatePipelineSteps() {
        let secondsElapsed = 0;
        resetPipelineSteps();
        setStepActive(STEPS[0].id, STEPS[0].label);

        return setInterval(() => {
            secondsElapsed++;

            if (secondsElapsed === 2) {
                setStepDone(STEPS[0].id);
                setStepActive(STEPS[1].id, STEPS[1].label);
            } else if (secondsElapsed === 4) {
                setStepDone(STEPS[1].id);
                setStepActive(STEPS[2].id, 'OCR / vision processing…');
            } else if (secondsElapsed > 4) {
                const mins = Math.floor(secondsElapsed / 60);
                const secs = secondsElapsed % 60;
                const timeStr = mins > 0 ? `${mins}m ${secs < 10 ? '0' : ''}${secs}s` : `${secs}s`;

                const hints = [
                    `Transcribing handwritten & scanned notes (${timeStr})…`,
                    `Analyzing document structure & formulas (${timeStr})…`,
                    `Vision OCR active across document pages (${timeStr})…`,
                    `Neural transcription in progress (${timeStr})…`
                ];
                const hint = hints[Math.floor(secondsElapsed / 3) % hints.length];
                if (processingCurrentStep) processingCurrentStep.textContent = hint;
            }
        }, 1000);
    }

    // ── File Selection & Session Listeners ──
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleUpload(e.target.files[0]);
        }
    });

    changeFileBtn.addEventListener('click', resetSession);

    if (navChat) {
        navChat.addEventListener('click', () => {
            if (activeDocumentId) {
                welcomeScreen.classList.add('hidden');
                chatSection.classList.remove('hidden');
                chatInput.focus();
            } else {
                showToast('Please upload a PDF document first.', 'info');
            }
        });
    }

    mobileSidebarToggle && mobileSidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('mobile-open');
    });

    // ── Handle Upload Pipeline ──
    async function handleUpload(file) {
        if (!file || isUploading) return;

        if (!file.name.toLowerCase().endsWith('.pdf')) {
            showToast('Please select a valid PDF file.', 'error');
            return;
        }

        isUploading = true;
        if (welcomeDropTarget) welcomeDropTarget.classList.add('hidden');
        if (processingContainer) processingContainer.classList.remove('hidden');
        resetPipelineSteps();

        const stepTimer = animatePipelineSteps();
        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData
            });

            clearInterval(stepTimer);

            if (!res.ok) {
                const errData = await res.json().catch(() => ({ detail: 'Upload failed.' }));
                throw new Error(errData.detail || 'Failed to process document.');
            }

            const data = await res.json();
            markAllPipelineDone();

            setTimeout(() => onUploadComplete(data), 600);

        } catch (err) {
            clearInterval(stepTimer);
            console.error('Upload Error:', err);
            showToast(err.message || 'Upload failed. Please try again.', 'error');
            if (processingContainer) processingContainer.classList.add('hidden');
            if (welcomeDropTarget) welcomeDropTarget.classList.remove('hidden');
            fileInput.value = '';
        } finally {
            isUploading = false;
        }
    }

    function onUploadComplete(data) {
        activeDocumentId = data.document_id;

        if (processingContainer) processingContainer.classList.add('hidden');
        if (docBlockSection) docBlockSection.classList.remove('hidden');
        if (documentInfoCard) documentInfoCard.classList.remove('hidden');
        if (sidebarEmptyState) sidebarEmptyState.classList.add('hidden');

        docName.textContent = data.filename || 'document.pdf';
        const ocrTag = data.ocr_pages > 0 ? ` • ✍ ${data.ocr_pages} OCR pages` : '';
        docStats.textContent = `${data.pages} pages • ${data.chunks} chunks${ocrTag}`;

        if (chatDocTitle) chatDocTitle.textContent = truncate(data.filename || 'Document', 32);

        // Render suggested questions
        renderSuggestedQuestions(data.suggested_questions);

        // Switch main view from welcome screen to chat
        welcomeScreen.classList.add('hidden');
        chatSection.classList.remove('hidden');

        if (window.innerWidth <= 900) {
            sidebar.classList.remove('mobile-open');
        }

        chatInput.focus();
    }

    function resetSession() {
        activeDocumentId = null;
        fileInput.value = '';
        conversationHistory = [];

        if (welcomeDropTarget) welcomeDropTarget.classList.remove('hidden');
        if (processingContainer) processingContainer.classList.add('hidden');
        if (docBlockSection) docBlockSection.classList.add('hidden');
        if (documentInfoCard) documentInfoCard.classList.add('hidden');
        if (questionsSection) questionsSection.classList.add('hidden');
        if (sidebarEmptyState) sidebarEmptyState.classList.remove('hidden');
        chatSection.classList.add('hidden');
        welcomeScreen.classList.remove('hidden');

        clearChatFeed();
    }

    function clearChatFeed(e) {
        if (e) {
            e.preventDefault();
            e.stopPropagation();
        }

        // Abort any ongoing stream
        if (activeAbortController) {
            try {
                activeAbortController.abort();
            } catch (_) {}
            activeAbortController = null;
        }

        isStreaming = false;
        conversationHistory = [];

        if (chatHistory) {
            chatHistory.innerHTML = `
                <div class="feed-welcome-banner" id="chatWelcomeMsg">
                    <div class="banner-icon">💡</div>
                    <div class="banner-body">
                        <p class="banner-title">Document indexed and ready</p>
                        <p class="banner-sub">Ask questions below or select a suggested topic from the sidebar.</p>
                    </div>
                </div>
            `;
            chatHistory.scrollTop = 0;
        }

        if (chatInput) {
            chatInput.value = '';
            chatInput.disabled = false;
            chatInput.focus();
        }

        if (sendBtn) {
            sendBtn.disabled = false;
        }

        showToast('Chat history cleared', 'info');
    }

    if (clearChatBtn) {
        clearChatBtn.onclick = clearChatFeed;
    }

    // ── Render Suggested Questions ──
    function renderSuggestedQuestions(questions) {
        questionsGrid.innerHTML = '';
        if (!questions || questions.length === 0) {
            questionsSection.classList.add('hidden');
            return;
        }

        questionsSection.classList.remove('hidden');

        questions.forEach(q => {
            const btn = document.createElement('button');
            btn.className = 'question-pill';
            btn.type = 'button';
            btn.setAttribute('aria-label', `Ask: ${q}`);
            btn.innerHTML = `
                <span class="question-pill-text">${escapeHtml(q)}</span>
                <span class="question-pill-arrow" aria-hidden="true">→</span>
            `;
            btn.addEventListener('click', () => {
                chatInput.value = q;
                submitQuery(q);
                if (window.innerWidth <= 900) sidebar.classList.remove('mobile-open');
            });
            questionsGrid.appendChild(btn);
        });
    }

    // ── Chat Submission & Real-Time Streaming ──
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const query = chatInput.value.trim();
        if (query && !isStreaming) submitQuery(query);
    });

    async function submitQuery(question) {
        if (!activeDocumentId) {
            showToast('Please upload a PDF document first.', 'warn');
            return;
        }
        if (!question.trim() || isStreaming) return;

        isStreaming = true;
        sendBtn.disabled = true;
        chatInput.disabled = true;

        // Hide banner
        const banner = document.getElementById('chatWelcomeMsg');
        if (banner) banner.style.display = 'none';

        // Render user message bubble
        appendUserMessage(question);
        chatInput.value = '';

        // Create AI message placeholder
        const msgId = 'msg-' + Date.now();
        const { bubbleEl } = appendAIMessagePlaceholder(msgId);

        // Blinking stream cursor
        const streamCursor = document.createElement('span');
        streamCursor.className = 'stream-cursor';
        streamCursor.id = `${msgId}-cursor`;

        let fullAnswer = '';
        let streamBuffer = '';
        let flowchartData = null;
        let sourcesData = null;
        let isRefusal = false;
        const refusalKey = "couldn't find the answer to that in the uploaded pdf";

        activeAbortController = new AbortController();

        try {
            const response = await fetch(`${API_BASE}/chat/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    document_id: activeDocumentId,
                    question: question,
                    history: conversationHistory.slice(-6)
                }),
                signal: activeAbortController.signal
            });

            if (!response.ok) {
                const errJson = await response.json().catch(() => ({ detail: 'Stream request failed.' }));
                throw new Error(errJson.detail || 'Error communicating with AI service.');
            }

            // Remove thinking indicator and create answer prose container
            const thinkingEl = document.getElementById(`${msgId}-thinking`);
            if (thinkingEl) thinkingEl.remove();

            const textContainer = document.createElement('div');
            textContainer.id = `${msgId}-text`;
            textContainer.className = 'answer-prose';
            bubbleEl.appendChild(textContainer);
            bubbleEl.appendChild(streamCursor);

            // Read SSE Stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                streamBuffer += decoder.decode(value, { stream: true });
                const lines = streamBuffer.split('\n');
                streamBuffer = lines.pop(); // Retain incomplete line

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const jsonStr = line.slice(6).trim();
                    if (!jsonStr) continue;

                    let event;
                    try {
                        event = JSON.parse(jsonStr);
                    } catch {
                        continue;
                    }

                    switch (event.type) {
                        case 'status':
                            updateThinkingText(msgId, event.message || '');
                            break;

                        case 'chunk':
                            // In case thinking element still exists
                            const think = document.getElementById(`${msgId}-thinking`);
                            if (think) think.remove();

                            fullAnswer += event.text || '';
                            textContainer.innerHTML = renderMarkdown(fullAnswer);
                            textContainer.appendChild(streamCursor);
                            scrollToBottom();
                            break;

                        case 'sources':
                            sourcesData = event.data;
                            if (fullAnswer.toLowerCase().includes(refusalKey)) {
                                isRefusal = true;
                            }
                            if (sourcesData && sourcesData.length > 0 && !isRefusal) {
                                renderCitations(bubbleEl, sourcesData);
                                scrollToBottom();
                            }
                            break;

                        case 'flowchart':
                            flowchartData = event.data;
                            if (fullAnswer.toLowerCase().includes(refusalKey)) {
                                isRefusal = true;
                            }
                            if (flowchartData && !isRefusal) {
                                await renderFlowchart(bubbleEl, flowchartData);
                                scrollToBottom();
                            }
                            break;

                        case 'done':
                            break;

                        case 'error':
                            throw new Error(event.message || 'Error occurred during streaming.');
                    }
                }
            }

        } catch (err) {
            if (err.name === 'AbortError') {
                console.log('Stream query aborted by user.');
                return;
            }
            console.error('Streaming error:', err);
            const think = document.getElementById(`${msgId}-thinking`);
            if (think) think.remove();

            if (!fullAnswer) {
                bubbleEl.innerHTML = `<span style="color: var(--color-accent); font-size: 0.85rem;">⚠️ ${escapeHtml(err.message || 'Failed to complete query.')}</span>`;
            }
        } finally {
            activeAbortController = null;
            // Remove cursor
            streamCursor.remove();

            // Finalize markdown rendering
            const textNode = document.getElementById(`${msgId}-text`);
            if (textNode && fullAnswer) {
                textNode.innerHTML = renderMarkdown(fullAnswer);
            }

            // Fallback render if events arrived without triggering render
            if (sourcesData && sourcesData.length > 0 && !bubbleEl.querySelector('.citations-section') && !isRefusal) {
                renderCitations(bubbleEl, sourcesData);
            }
            if (flowchartData && !bubbleEl.querySelector('.diagram-section') && !isRefusal) {
                await renderFlowchart(bubbleEl, flowchartData);
            }

            // Save conversation history on valid answer
            if (fullAnswer && !isRefusal) {
                conversationHistory.push({ role: 'user', content: question });
                conversationHistory.push({ role: 'assistant', content: fullAnswer });
            }

            isStreaming = false;
            sendBtn.disabled = false;
            chatInput.disabled = false;
            chatInput.focus();
            scrollToBottom();
        }
    }

    // ── Render User Message ──
    function appendUserMessage(text) {
        const row = document.createElement('div');
        row.className = 'chat-msg-row user';
        row.innerHTML = `
            <div class="bubble-wrap">
                <span class="bubble-sender">You</span>
                <div class="chat-bubble">${escapeHtml(text)}</div>
            </div>
            <div class="avatar-badge">You</div>
        `;
        chatHistory.appendChild(row);
        scrollToBottom();
    }

    // ── Render AI Message Placeholder ──
    function appendAIMessagePlaceholder(id) {
        const row = document.createElement('div');
        row.className = 'chat-msg-row ai';
        row.id = id;

        row.innerHTML = `
            <div class="avatar-badge">AI</div>
            <div class="bubble-wrap">
                <span class="bubble-sender">QueryFolio AI</span>
                <div class="chat-bubble" id="${id}-bubble">
                    <div class="thinking-box" id="${id}-thinking">
                        <div class="thinking-pulse-dots" aria-hidden="true">
                            <span class="pulse-dot"></span>
                            <span class="pulse-dot"></span>
                            <span class="pulse-dot"></span>
                        </div>
                        <span class="thinking-label" id="${id}-thinking-label">Searching document…</span>
                    </div>
                </div>
            </div>
        `;
        chatHistory.appendChild(row);
        scrollToBottom();

        const bubbleEl = document.getElementById(`${id}-bubble`);
        return { row, bubbleEl };
    }

    function updateThinkingText(msgId, text) {
        const label = document.getElementById(`${msgId}-thinking-label`);
        if (label) label.textContent = text;
    }

    // ── Mermaid Flowchart Renderer ──
    async function renderFlowchart(bubbleEl, data) {
        if (!data || !data.definition) return;

        const section = document.createElement('div');
        section.className = 'diagram-section';
        const diagramId = 'mermaid-' + Date.now() + '-' + Math.random().toString(36).substring(2, 7);

        section.innerHTML = `
            <div class="diagram-header-bar">
                <span class="diagram-badge">
                    <span class="diagram-badge-dot"></span>
                    Visual Explanation Flowchart
                </span>
                <button type="button" class="diagram-copy-btn" id="copy-${diagramId}" title="Copy Mermaid Diagram Code">
                    Copy Diagram
                </button>
            </div>
            <div class="diagram-viewport" id="viewport-${diagramId}">
                <div class="spinner-ring" style="width: 18px; height: 18px; border: 2px solid rgba(99,102,241,0.2); border-top-color: var(--brand-indigo); border-radius: 50%; animation: spinRing 0.8s linear infinite;"></div>
            </div>
            ${data.description ? `
            <div class="diagram-description-card" id="desc-${diagramId}">
                <div class="diagram-description-title">
                    <span class="diagram-desc-icon">📖</span>
                    <span>Flowchart Walkthrough &amp; Description (Simple English)</span>
                </div>
                <div class="diagram-description-body">
                    ${renderMarkdown(data.description)}
                </div>
            </div>` : ''}
        `;
        bubbleEl.appendChild(section);

        const viewport = document.getElementById(`viewport-${diagramId}`);
        const copyBtn  = document.getElementById(`copy-${diagramId}`);

        copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(data.definition).then(() => {
                copyBtn.textContent = 'Copied! ✓';
                setTimeout(() => { copyBtn.textContent = 'Copy Diagram'; }, 2000);
            });
        });

        try {
            if (typeof window.mermaid === 'undefined') {
                throw new Error('Mermaid renderer library not available.');
            }

            const cleanDef = sanitizeMermaidDefinition(data.definition);
            if (!cleanDef) throw new Error('Invalid diagram definition structure.');

            const { svg } = await window.mermaid.render(diagramId, cleanDef);
            viewport.innerHTML = svg;

        } catch (err) {
            console.warn('Mermaid rendering issue:', err);
            viewport.innerHTML = `
                <div class="diagram-error-card">
                    ⚠ Diagram visualization could not be generated.
                </div>
            `;
        }

        scrollToBottom();
    }

    function sanitizeMermaidDefinition(def) {
        if (!def || typeof def !== 'string') return '';
        let clean = def
            .replace(/<script[\s\S]*?<\/script>/gi, '')
            .replace(/<[^>]+>/g, '')
            .replace(/javascript:/gi, '')
            .replace(/on\w+\s*=/gi, '');

        clean = clean.trim();
        if (!clean.toLowerCase().startsWith('flowchart') && !clean.toLowerCase().startsWith('graph')) {
            clean = 'flowchart TD\n' + clean;
        }
        return clean;
    }

    // ── Verified Sources / Citations Renderer ──
    function renderCitations(bubbleEl, sources) {
        if (!sources || sources.length === 0) return;

        const section = document.createElement('div');
        section.className = 'citations-section';

        const toggle = document.createElement('div');
        toggle.className = 'citations-toggle';
        toggle.setAttribute('role', 'button');
        toggle.setAttribute('tabindex', '0');
        toggle.setAttribute('aria-expanded', 'false');
        toggle.innerHTML = `
            <span class="citations-toggle-label">
                📚 Verified Evidence
                <span class="citations-count-pill">${sources.length}</span>
            </span>
            <span class="citations-chevron" aria-hidden="true">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="6 9 12 15 18 9"/>
                </svg>
            </span>
        `;

        const container = document.createElement('div');
        container.className = 'citations-container hidden';

        sources.forEach(src => {
            const card = document.createElement('div');
            card.className = 'citation-card';

            const secTitle = src.section && src.section !== 'General'
                ? escapeHtml(src.section)
                : '';

            let ocrBadge = '';
            const isHw = src.ocr_derived || src.source_type === 'handwritten_ocr';
            if (isHw) {
                const isLow = src.low_confidence;
                const confPercent = src.ocr_confidence ? ` (${Math.round(src.ocr_confidence)}%)` : '';
                const methodLabel = src.extraction_method === 'vision_ocr' ? 'Vision OCR' : 'Handwritten/OCR';
                ocrBadge = `
                    <span class="ocr-source-badge${isLow ? ' low-conf' : ''}" title="Source extracted via ${escapeHtml(src.extraction_method || 'Vision OCR')}">
                        ✍ ${isLow ? 'Handwritten · Low Confidence' : methodLabel}${confPercent}
                    </span>
                `;
            }

            card.innerHTML = `
                <div class="citation-header-row">
                    <span class="source-page-badge">Page ${src.page}</span>
                    ${secTitle ? `<span class="source-section-pill">${secTitle}</span>` : ''}
                    ${ocrBadge}
                </div>
                <p class="citation-excerpt">"${escapeHtml(src.text || '')}"</p>
            `;
            container.appendChild(card);
        });

        section.appendChild(toggle);
        section.appendChild(container);
        bubbleEl.appendChild(section);

        function toggleAccordion() {
            const isOpen = !container.classList.contains('hidden');
            container.classList.toggle('hidden', isOpen);
            toggle.classList.toggle('expanded', !isOpen);
            toggle.setAttribute('aria-expanded', String(!isOpen));
            scrollToBottom();
        }

        toggle.addEventListener('click', toggleAccordion);
        toggle.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                toggleAccordion();
            }
        });
    }

    // ── Markdown Parser with Fallback ──
    function renderMarkdown(text) {
        if (!text) return '';
        if (typeof window.marked !== 'undefined' && typeof window.DOMPurify !== 'undefined') {
            try {
                window.marked.setOptions({ gfm: true, breaks: true });
                const parsed = window.marked.parse(text);
                return window.DOMPurify.sanitize(parsed, {
                    ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'b', 'i', 'h1', 'h2', 'h3', 'h4',
                                   'ul', 'ol', 'li', 'code', 'pre', 'blockquote', 'hr', 'span', 'div'],
                    ALLOWED_ATTR: ['class']
                });
            } catch {}
        }
        return safeFallbackMarkdown(text);
    }

    function safeFallbackMarkdown(text) {
        let s = escapeHtml(text);
        s = s.replace(/^### (.*)$/gim, '<h3>$1</h3>');
        s = s.replace(/^## (.*)$/gim, '<h2>$1</h2>');
        s = s.replace(/^# (.*)$/gim, '<h1>$1</h1>');
        s = s.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        s = s.replace(/\*(.*?)\*/g, '<em>$1</em>');
        return s.split('\n\n').map(p => `<p>${p.trim()}</p>`).join('');
    }

    // ── Utility Helpers ──
    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function truncate(str, max) {
        if (!str) return '';
        return str.length > max ? str.slice(0, max - 1) + '…' : str;
    }

    function showToast(message, type = 'info') {
        const old = document.getElementById('app-toast');
        if (old) old.remove();

        const toast = document.createElement('div');
        toast.id = 'app-toast';

        const borderColors = {
            error: '#f43f5e',
            warn: '#f59e0b',
            info: '#6366f1',
            success: '#10b981'
        };

        Object.assign(toast.style, {
            position: 'fixed',
            top: '20px',
            left: '50%',
            transform: 'translateX(-50%)',
            background: '#0d131f',
            border: `1px solid ${borderColors[type] || borderColors.info}`,
            color: '#f1f5f9',
            padding: '10px 22px',
            borderRadius: '100px',
            fontSize: '0.85rem',
            fontWeight: '500',
            zIndex: '99999',
            boxShadow: '0 8px 30px rgba(0,0,0,0.6)',
            animation: 'msgSlideIn 0.25s ease'
        });

        toast.textContent = message;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 4500);
    }
});
