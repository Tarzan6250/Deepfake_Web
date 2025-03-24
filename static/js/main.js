document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.querySelector('.preview-container');
    const preview = document.getElementById('preview');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resetBtn = document.getElementById('resetBtn');
    const resultContainer = document.querySelector('.result-container');
    const resultText = document.getElementById('resultText');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');
    const loading = document.getElementById('loading');
    const errorText = document.getElementById('errorText');

    function showElement(element) {
        if (element === loading) {
            element.style.display = 'flex';
        } else {
            element.style.display = 'block';
        }
    }

    function hideElement(element) {
        element.style.display = 'none';
    }

    function showError(message) {
        errorText.textContent = message;
        showElement(errorText);
    }

    function hideError() {
        hideElement(errorText);
    }

    function showLoading() {
        showElement(loading);
    }

    function hideLoading() {
        hideElement(loading);
    }

    function showResult(result) {
        const isReal = result.result.toLowerCase() === 'real';
        const resultClass = isReal ? 'success' : 'danger';
        const resultIcon = isReal ? '✓' : '✗';
        
        resultText.innerHTML = `
            <h2 class="${resultClass}" style="font-size: 2em; margin-bottom: 0.5em;">
                ${resultIcon} ${result.result.toUpperCase()}
            </h2>
            <p style="font-size: 1.2em; margin-bottom: 1em;">
                This ${result.type} is ${isReal ? 'authentic' : 'likely manipulated'}.
            </p>
        `;

        confidenceBar.style.width = `${result.confidence}%`;
        confidenceBar.style.backgroundColor = isReal ? 'var(--success-color)' : 'var(--danger-color)';
        confidenceText.textContent = `${Math.round(result.confidence)}% Confidence`;
        
        showElement(resultContainer);
    }

    // Make sure loading is hidden initially
    hideElement(loading);
    hideElement(previewContainer);
    hideElement(resultContainer);
    hideElement(errorText);
    showElement(dropZone);

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
        handleFiles(e.dataTransfer.files);
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        const fileName = file.name.toLowerCase();
        const fileExtension = fileName.split('.').pop();
        
        // Check file extension instead of type
        const allowedExtensions = ['jpg', 'jpeg', 'png', 'mp4', 'avi'];
        if (!allowedExtensions.includes(fileExtension)) {
            alert('Please upload a valid image (JPG, PNG) or video (MP4, AVI) file');
            return;
        }

        // Show preview
        const reader = new FileReader();
        reader.onload = function(e) {
            if (['jpg', 'jpeg', 'png'].includes(fileExtension)) {
                preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
            } else {
                preview.innerHTML = `
                    <video src="${e.target.result}" controls>
                        Your browser does not support the video tag.
                    </video>`;
            }
        }
        reader.readAsDataURL(file);

        showElement(previewContainer);
        hideElement(resultContainer);
        hideElement(dropZone);
    }

    async function analyzeFile() {
        const file = fileInput.files[0];
        const maxSize = 100 * 1024 * 1024; // 100MB in bytes

        if (!file) {
            showError('Please select a file first');
            return;
        }

        if (file.size > maxSize) {
            showError(`File size (${(file.size / (1024 * 1024)).toFixed(2)}MB) exceeds maximum limit of 100MB`);
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            showLoading();
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                if (response.status === 413) {
                    throw new Error('File size too large. Maximum allowed size is 100MB');
                }
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            if (result.error) {
                throw new Error(result.error);
            }
            showResult(result);
        } catch (error) {
            console.error('Error:', error);
            showError(`Error analyzing file: ${error.message}`);
        } finally {
            hideLoading();
        }
    }

    analyzeBtn.addEventListener('click', async () => {
        hideError();
        await analyzeFile();
    });

    resetBtn.addEventListener('click', () => {
        fileInput.value = '';
        preview.innerHTML = '';
        hideElement(previewContainer);
        hideElement(resultContainer);
        hideElement(loading);
        hideElement(errorText);
        showElement(dropZone);
    });
});
