document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const processingContainer = document.getElementById('processingContainer');
    const processingCurrentStep = document.getElementById('processingCurrentStep');
    const documentInfoCard = document.getElementById('documentInfoCard');
    const docName = document.getElementById('docName');
    const docStats = document.getElementById('docStats');
    const changeFileBtn = document.getElementById('changeFileBtn');
    
    const questionsSection = document.getElementById('questionsSection');
    const questionsGrid = document.getElementById('questionsGrid');
    
    const chatSection = document.getElementById('chat-section') || document.getElementById('chatSection');

    const chatHistory = document.getElementById('chatHistory');
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const chatWelcomeMsg = document.getElementById('chatWelcomeMsg');
    
    const appStatusBadge = document.getElementById('appStatusBadge');
    const appStatusText = document.getElementById('appStatusText');

    // Global App State
    let activeDocumentId = null;
    let isProcessing = false;

    // API Base URL: Uses relative '/api' if hosted directly by FastAPI (:8000), otherwise points to local backend server
    const API_BASE = (window.location.port === '8000') ? '/api' : 'http://127.0.0.1:8000/api';


    // Steps array for progress visualizer
    const steps = [
        { id: 'step-upload', label: 'Uploading PDF...' },
        { id: 'step-extract', label: 'Extracting text...' },
        { id: 'step-chunk', label: 'Creating chunks...' },
        { id: 'step-embed', label: 'Generating embeddings...' },
        { id: 'step-index', label: 'Building vector index...' },
        { id: 'step-topics', label: 'Finding important topics...' },
        { id: 'step-questions', label: 'Generating suggested questions...' }
    ];

    // File Input Click
    browseBtn.addEventListener('click', () => fileInput.click());

    // Drag & Drop Events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Change File Reset
    changeFileBtn.addEventListener('click', () => {
        activeDocumentId = null;
        fileInput.value = '';
        dropZone.classList.remove('hidden');
        processingContainer.classList.add('hidden');
        documentInfoCard.classList.add('hidden');
        questionsSection.classList.add('hidden');
        chatSection.classList.add('hidden');
        
        // Reset status badge
        appStatusBadge.classList.remove('active');
        appStatusText.textContent = 'Ready for Upload';
        
        // Reset Chat History
        chatHistory.innerHTML = `
            <div class="welcome-chat-msg" id="chatWelcomeMsg">
                <p>💡 Ask any question about your document or select an <strong>Important Question</strong> above to start.</p>
            </div>
        `;
    });

    // Handle File Upload
    async function handleFileUpload(file) {
        if (!file) return;

        if (!file.name.toLowerCase().endswith?.('.pdf') && !file.name.toLowerCase().endsWith('.pdf')) {
            alert('Please select a valid PDF file (.pdf extension).');
            return;
        }

        if (isProcessing) return;
        isProcessing = true;

        // UI Reset & Show Progress
        dropZone.classList.add('hidden');
        processingContainer.classList.remove('hidden');
        resetProgressSteps();

        // Simulate step progression visually during API upload request
        const stepInterval = runVisualProgressSteps();

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData
            });

            clearInterval(stepInterval);

            if (!response.ok) {
                const errData = await response.json().catch(() => ({ detail: 'Upload failed.' }));
                throw new Error(errData.detail || 'Failed to upload PDF.');
            }

            const data = await response.json();
            
            // Mark all progress steps completed
            markAllStepsCompleted();

            setTimeout(() => {
                onUploadSuccess(data);
            }, 600);

        } catch (error) {
            clearInterval(stepInterval);
            console.error('Upload Error:', error);
            alert(error.message || 'An error occurred during file upload. Please try again.');
            
            // Reset UI
            processingContainer.classList.add('hidden');
            dropZone.classList.remove('hidden');
            fileInput.value = '';
        } finally {
            isProcessing = false;
        }
    }

    // Step Progress Animation Helper
    function runVisualProgressSteps() {
        let currentIdx = 0;
        markStepActive(steps[0].id, steps[0].label);

        return setInterval(() => {
            if (currentIdx < steps.length - 1) {
                markStepCompleted(steps[currentIdx].id);
                currentIdx++;
                markStepActive(steps[currentIdx].id, steps[currentIdx].label);
            }
        }, 1200);
    }

    function resetProgressSteps() {
        steps.forEach(step => {
            const el = document.getElementById(step.id);
            if (el) {
                el.className = 'step-item';
            }
        });
    }

    function markStepActive(stepId, label) {
        const el = document.getElementById(stepId);
        if (el) {
            el.className = 'step-item active';
        }
        if (processingCurrentStep) {
            processingCurrentStep.textContent = label;
        }
    }

    function markStepCompleted(stepId) {
        const el = document.getElementById(stepId);
        if (el) {
            el.className = 'step-item completed';
        }
    }

    function markAllStepsCompleted() {
        steps.forEach(step => markStepCompleted(step.id));
        if (processingCurrentStep) {
            processingCurrentStep.textContent = '✓ Document processed successfully!';
        }
    }

    // On Upload Success UI Update
    function onUploadSuccess(data) {
        activeDocumentId = data.document_id;
        
        // Hide processing, show doc info card
        processingContainer.classList.add('hidden');
        documentInfoCard.classList.remove('hidden');

        docName.textContent = data.filename;
        docStats.textContent = `${data.pages} pages • ${data.chunks} text chunks`;

        // Update status badge
        appStatusBadge.classList.add('active');
        appStatusText.textContent = 'Document Ready';

        // Render Suggested Questions
        renderSuggestedQuestions(data.suggested_questions);

        // Show sections
        questionsSection.classList.remove('hidden');
        chatSection.classList.remove('hidden');
    }

    // Render Suggested Questions Pills
    function renderSuggestedQuestions(questions) {
        questionsGrid.innerHTML = '';

        if (!questions || questions.length === 0) {
            questionsSection.classList.add('hidden');
            return;
        }

        questions.forEach(qText => {
            const pill = document.createElement('button');
            pill.className = 'question-pill';
            pill.type = 'button';
            pill.innerHTML = `
                <span>${escapeHtml(qText)}</span>
                <span class="pill-arrow">→</span>
            `;

            pill.addEventListener('click', () => {
                // Ensure chat section is visible
                chatSection.classList.remove('hidden');

                // Smooth scroll to chat section
                const targetSec = document.getElementById('chat-section') || document.getElementById('chatSection') || chatSection;
                if (targetSec) {
                    targetSec.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }

                // Put question into input and focus
                chatInput.value = qText;
                chatInput.focus();

                // Submit question directly
                submitChatQuestion(qText);
            });

            questionsGrid.appendChild(pill);
        });
    }

    // Chat Form Submit
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const question = chatInput.value.trim();
        if (question) {
            submitChatQuestion(question);
        }
    });

    // Submit Chat Question
    async function submitChatQuestion(question) {
        if (!activeDocumentId) {
            alert('Please upload a PDF first.');
            return;
        }

        if (!question.trim()) return;

        // Ensure chat section is visible & scrolled into view
        chatSection.classList.remove('hidden');
        const targetSec = document.getElementById('chat-section') || document.getElementById('chatSection') || chatSection;
        if (targetSec) {
            targetSec.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }

        // Focus chat input
        chatInput.focus();

        // Hide welcome message if present
        const welcomeMsg = document.getElementById('chatWelcomeMsg');
        if (welcomeMsg) welcomeMsg.style.display = 'none';

        // Append User Message
        appendUserMessage(question);

        // Clear Input
        chatInput.value = '';

        // Append Thinking Indicator with step progression
        const thinkingId = appendThinkingIndicator();

        try {
            const response = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    document_id: activeDocumentId,
                    question: question
                })
            });

            removeThinkingIndicator(thinkingId);

            if (!response.ok) {
                const errData = await response.json().catch(() => ({ detail: 'Failed to get answer.' }));
                throw new Error(errData.detail || 'Error communicating with AI service.');
            }

            const data = await response.json();
            appendAIMessage(data.answer, data.sources, data.debug_info);

        } catch (error) {
            removeThinkingIndicator(thinkingId);
            console.error('Chat Error:', error);
            appendAIMessage(`⚠️ ${error.message || 'The AI service is temporarily unavailable. Please try again.'}`, []);
        }
    }

    // Append User Message to Chat
    function appendUserMessage(text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'chat-message user';
        msgDiv.innerHTML = `
            <span class="msg-sender">You</span>
            <div class="msg-bubble">${escapeHtml(text)}</div>
        `;
        chatHistory.appendChild(msgDiv);
        scrollToBottom();
    }

    // Append Thinking Indicator with state progression
    function appendThinkingIndicator() {
        const id = 'thinking-' + Date.now();
        const div = document.createElement('div');
        div.className = 'chat-message ai';
        div.id = id;
        div.innerHTML = `
            <span class="msg-sender">AI</span>
            <div class="thinking-indicator">
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="thinking-text" id="${id}-text">Searching the document...</span>
            </div>
        `;
        chatHistory.appendChild(div);
        scrollToBottom();

        // Dynamically transition loading state
        setTimeout(() => {
            const txtEl = document.getElementById(`${id}-text`);
            if (txtEl) {
                txtEl.textContent = 'Generating answer...';
            }
        }, 1200);

        return id;
    }

    function removeThinkingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    // Format Markdown safely to sanitized HTML
    function renderFormattedMarkdown(text) {
        if (!text) return '';

        // Primary rendering via CDN libraries if available
        if (typeof window.marked !== 'undefined' && typeof window.DOMPurify !== 'undefined') {
            try {
                window.marked.setOptions({ gfm: true, breaks: true });
                const rawHtml = window.marked.parse(text);
                return window.DOMPurify.sanitize(rawHtml);
            } catch (err) {
                console.warn('Marked parsing error, using fallback:', err);
            }
        }

        // Robust pure-JS fallback markdown parser
        return fallbackMarkdownParser(text);
    }

    function fallbackMarkdownParser(text) {
        if (!text) return '';
        let escaped = escapeHtml(text);

        // Headings
        escaped = escaped.replace(/^### (.*$)/gim, '<h3 class="answer-section-heading">$1</h3>');
        escaped = escaped.replace(/^## (.*$)/gim, '<h2 class="answer-section-heading">$1</h2>');
        escaped = escaped.replace(/^# (.*$)/gim, '<h1 class="answer-heading">$1</h1>');

        // Bold & Italics
        escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        escaped = escaped.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // Lists & Paragraphs
        const lines = escaped.split('\n');
        let inList = false;
        let inOrderedList = false;
        const result = [];

        for (let line of lines) {
            const listMatch = line.match(/^[\s]*[\*\-]\s+(.*)/);
            const numMatch = line.match(/^[\s]*\d+\.\s+(.*)/);

            if (listMatch) {
                if (!inList) {
                    result.push('<ul class="answer-list">');
                    inList = true;
                }
                result.push(`<li>${listMatch[1]}</li>`);
            } else if (numMatch) {
                if (!inOrderedList) {
                    result.push('<ol class="answer-list">');
                    inOrderedList = true;
                }
                result.push(`<li>${numMatch[1]}</li>`);
            } else {
                if (inList) {
                    result.push('</ul>');
                    inList = false;
                }
                if (inOrderedList) {
                    result.push('</ol>');
                    inOrderedList = false;
                }
                const trimmed = line.trim();
                if (trimmed) {
                    if (!trimmed.startsWith('<h') && !trimmed.startsWith('<ul') && !trimmed.startsWith('<ol')) {
                        result.push(`<p class="answer-paragraph">${trimmed}</p>`);
                    } else {
                        result.push(trimmed);
                    }
                }
            }
        }

        if (inList) result.push('</ul>');
        if (inOrderedList) result.push('</ol>');

        return result.join('\n');
    }

    // Append AI Message & Section-Aware Source References
    function appendAIMessage(answerText, sources, debugInfo = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'chat-message ai';

        let sourcesHtml = '';
        if (sources && sources.length > 0) {
            const sourcesItems = sources.map(s => {
                const sectionLabel = s.section ? ` · ${escapeHtml(s.section)}` : '';
                return `
                <div class="source-item">
                    <div class="source-page-badge">📄 Page ${s.page}${sectionLabel}</div>
                    <div class="source-excerpt">"${escapeHtml(s.text)}"</div>
                </div>
            `;
            }).join('');

            sourcesHtml = `
                <div class="sources-container">
                    <div class="sources-title" onclick="this.nextElementSibling.classList.toggle('hidden')">
                        <span>📚 Sources (${sources.length} section passages referenced)</span>
                        <span style="font-size:0.7rem; opacity:0.8;">[toggle]</span>
                    </div>
                    <div class="sources-list hidden">
                        ${sourcesItems}
                    </div>
                </div>
            `;
        }

        let debugHtml = '';
        if (debugInfo && debugInfo.initial_candidates_count) {
            debugHtml = `
                <div class="debug-indicator-badge" style="font-size: 0.72rem; color: #a0aec0; margin-top: 6px; padding: 3px 8px; background: rgba(255,255,255,0.06); border-radius: 4px; display: inline-block;">
                    ⚡ Retrieval: ${debugInfo.initial_candidates_count} candidates → ${debugInfo.final_ranked_count} after reranking
                </div>
            `;
        }

        const formattedAnswerHtml = renderFormattedMarkdown(answerText);

        msgDiv.innerHTML = `
            <span class="msg-sender">AI</span>
            <div class="msg-bubble">
                ${formattedAnswerHtml}
                ${sourcesHtml}
                ${debugHtml}
            </div>
        `;

        chatHistory.appendChild(msgDiv);
        scrollToBottom();
    }


    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
