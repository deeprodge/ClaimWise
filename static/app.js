document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileUpload');
    const uploadedFiles = document.getElementById('uploadedFiles');
    const chatMessages = document.getElementById('chatMessages');
    const userQuery = document.getElementById('userQuery');
    const queryButton = document.getElementById('queryButton');

    // Add at the top of the file, inside DOMContentLoaded
    let isProcessing = false;

    function showProcessingState(show) {
        const uploadArea = document.getElementById('dropZone');
        if (show) {
            uploadArea.classList.add('processing');
            uploadArea.innerHTML = `
                <div class="loading-spinner"></div>
                <p>Processing document...</p>
                <small>This may take a few moments</small>
            `;
        } else {
            uploadArea.classList.remove('processing');
            uploadArea.innerHTML = `
                <input type="file" id="fileUpload" accept=".pdf,.txt" multiple />
                <p>Drag & drop files or click to upload</p>
                <small>Supported: PDF, TXT</small>
            `;
            // Reattach file input listener
            document.getElementById('fileUpload').addEventListener('change', () => {
                handleFiles(fileInput.files);
            });
        }
    }

    // Drag and drop handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        handleFiles(files);
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        handleFiles(fileInput.files);
    });

    async function handleFiles(files) {
        for (const file of files) {
            await uploadFile(file);
        }
    }

    async function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            showProcessingState(true);
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.success) {
                addFileToList(file.name);
                addMessage('Successfully uploaded and processed ' + file.name, 'system');
            } else {
                addMessage('Failed to process ' + file.name + ': ' + data.message, 'system');
            }
        } catch (error) {
            console.error('Error:', error);
            addMessage('Error processing file', 'system');
        } finally {
            showProcessingState(false);
        }
    }

    function addFileToList(filename) {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.textContent = filename;
        uploadedFiles.appendChild(fileItem);
    }

    function addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        // Convert markdown to HTML
        const formattedContent = content
            // Bold text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Italic text
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // New lines
            .replace(/\n/g, '<br>')
            // Lists
            .replace(/- (.*)/g, '• $1')
            // Headers
            .replace(/### (.*)/g, '<h3>$1</h3>')
            .replace(/## (.*)/g, '<h2>$1</h2>')
            .replace(/# (.*)/g, '<h1>$1</h1>');

        // Use innerHTML for formatted content
        messageDiv.innerHTML = formattedContent;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function addTypingIndicator() {
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'message system typing-indicator';
        typingIndicator.innerHTML = '<em>ClaimWise is typing...</em>';
        chatMessages.appendChild(typingIndicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return typingIndicator;
    }

    function removeTypingIndicator(typingIndicator) {
        if (typingIndicator) {
            chatMessages.removeChild(typingIndicator);
        }
    }

    async function sendQuery() {
        const query = userQuery.value.trim();
        if (!query) return;

        addMessage(query, 'user');
        userQuery.value = '';

        const typingIndicator = addTypingIndicator();

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query })
            });

            const data = await response.json();
            removeTypingIndicator(typingIndicator);

            if (data.success) {
                addMessage(data.response, 'system');
            } else {
                addMessage('Sorry, I couldn\'t process your query.', 'system');
                if (data.error) {
                    console.error('Error details:', data.error);
                    addMessage('Error: ' + data.error, 'error');
                }
            }
        } catch (error) {
            console.error('Error:', error);
            removeTypingIndicator(typingIndicator);
            addMessage('Sorry, there was an error processing your query.', 'system');
        }
    }

    async function resetDocuments() {
        try {
            showProcessingState(true);
            const response = await fetch('/reset', {
                method: 'POST'
            });
            const data = await response.json();
            
            if (data.success) {
                // Clear uploaded files list
                uploadedFiles.innerHTML = '';
                addMessage('All documents have been cleared.', 'system');
            } else {
                addMessage('Failed to reset documents: ' + data.message, 'error');
            }
        } catch (error) {
            console.error('Error:', error);
            addMessage('Error resetting documents', 'error');
        } finally {
            showProcessingState(false);
        }
    }

    // Event listeners for sending messages
    queryButton.addEventListener('click', sendQuery);
    userQuery.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    });

    // Add reset button handler
    const resetButton = document.getElementById('resetButton');
    if (resetButton) {
        resetButton.addEventListener('click', resetDocuments);
    }
});
