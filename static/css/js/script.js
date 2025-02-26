document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const indexBtn = document.getElementById('indexBtn');
    const indexStatus = document.getElementById('indexStatus');
    const searchQuery = document.getElementById('searchQuery');
    const searchBtn = document.getElementById('searchBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsSection = document.getElementById('resultsSection');
    const searchResults = document.getElementById('searchResults');
    const chatToggle = document.getElementById('chatToggle');
    const chatbotCard = document.getElementById('chatbotCard');
    const minimizeChat = document.getElementById('minimizeChat');
    const chatMessages = document.getElementById('chatMessages');
    const chatInput = document.getElementById('chatInput');
    const sendMessage = document.getElementById('sendMessage');
    
    // Hide loading indicator initially
    loadingIndicator.style.display = 'none';
    
    // Current grade level (can be changed through UI)
    let currentGradeLevel = null;
    
    // Update grade level
    const gradeButtons = document.querySelectorAll('.grade-pill');
    gradeButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons
            gradeButtons.forEach(btn => btn.classList.remove('active-grade'));
            // Add active class to clicked button
            this.classList.add('active-grade');
            // Update current grade level
            currentGradeLevel = this.textContent;
        });
    });
    
    // Index PDFs
    indexBtn.addEventListener('click', function() {
        indexBtn.disabled = true;
        indexStatus.innerHTML = '<div class="spinner-border spinner-border-sm" role="status"><span class="visually-hidden">Loading...</span></div> Indexing PDFs...';
        
        fetch('/index-pdfs', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                indexStatus.innerHTML = `<div class="alert alert-success">${data.message}</div>`;
            } else {
                indexStatus.innerHTML = `<div class="alert alert-danger">Error: ${data.message}</div>`;
            }
            indexBtn.disabled = false;
        })
        .catch(error => {
            indexStatus.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            indexBtn.disabled = false;
        });
    });
    
    // Search functionality
    searchBtn.addEventListener('click', performSearch);
    searchQuery.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
    
    function performSearch() {
        const query = searchQuery.value.trim();
        if (!query) return;
        
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        resultsSection.style.display = 'none';
        
        fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                grade_level: currentGradeLevel
            })
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading indicator
            loadingIndicator.style.display = 'none';
            
            if (data.status === 'success') {
                displayResults(data.formatted_response, query);
                resultsSection.style.display = 'block';
            } else {
                searchResults.innerHTML = `<div class="alert alert-danger">Error: ${data.message}</div>`;
                resultsSection.style.display = 'block';
            }
        })
        .catch(error => {
            loadingIndicator.style.display = 'none';
            searchResults.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            resultsSection.style.display = 'block';
        });
    }
    
    function displayResults(formattedResponse, query) {
        const answer = formattedResponse.answer;
        const sources = formattedResponse.sources;
        const relatedQuestions = formattedResponse.related_questions || [];
        
        let html = `
            <div class="card mb-4">
                <div class="card-header bg-primary text-white">
                    <i class="fas fa-graduation-cap me-2"></i> Study Answer
                </div>
                <div class="card-body">
                    <h5 class="card-title">${query}</h5>
                    <div class="answer-content">
                        ${answer.split('\n').map(para => `<p>${para}</p>`).join('')}
                    </div>
                </div>
            </div>
        `;
        
        // Sources card
        html += `
            <div class="card mb-4">
                <div class="card-header">
                    <i class="fas fa-book me-2"></i> Sources
                </div>
                <div class="card-body">
                    <ul class="list-group">
        `;
        
        if (sources.length > 0) {
            sources.forEach(source => {
                html += `
                    <li class="list-group-item">
                        <div><strong>${source.book}</strong> (Page ${source.page})</div>
                        <div class="text-muted small">${source.excerpt}</div>
                    </li>
                `;
            });
        } else {
            html += `<li class="list-group-item">No specific sources found</li>`;
        }
        
        html += `
                    </ul>
                </div>
            </div>
        `;
        
        // Related questions
        if (relatedQuestions.length > 0) {
            html += `
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-question-circle me-2"></i> Related Questions
                    </div>
                    <div class="card-body">
                        <div class="d-flex flex-wrap">
            `;
            
            relatedQuestions.forEach(question => {
                html += `
                    <div class="related-question me-2 mb-2" onclick="document.getElementById('searchQuery').value = this.textContent; document.getElementById('searchBtn').click();">
                        ${question}
                    </div>
                `;
            });
            
            html += `
                        </div>
                    </div>
                </div>
            `;
        }
        
        searchResults.innerHTML = html;
    }
    
    // Chat functionality
    chatToggle.addEventListener('click', function() {
        chatbotCard.classList.toggle('show-chat');
    });
    
    minimizeChat.addEventListener('click', function() {
        chatbotCard.classList.remove('show-chat');
    });
    
    function sendChatMessage() {
        const message = chatInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        chatMessages.innerHTML += `
            <div class="chat-message user-message">
                ${message}
            </div>
        `;
        
        // Clear input
        chatInput.value = '';
        
        // Add typing indicator
        const typingId = 'typing-' + Date.now();
        chatMessages.innerHTML += `
            <div class="chat-message bot-message" id="${typingId}">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Send to backend
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                grade_level: currentGradeLevel
            })
        })
        .then(response => response.json())
        .then(data => {
            // Remove typing indicator
            document.getElementById(typingId).remove();
            
            if (data.status === 'success') {
                const response = data.response;
                
                // Add bot response
                chatMessages.innerHTML += `
                    <div class="chat-message bot-message">
                        ${response.answer.split('\n').map(para => `<p>${para}</p>`).join('')}
                    </div>
                `;
                
                // Add related questions as suggestions
                if (response.related_questions && response.related_questions.length > 0) {