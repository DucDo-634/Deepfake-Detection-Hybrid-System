let currentFile = null;

// ================= AUTOMATIC POPUP LOGIC =================
document.addEventListener('DOMContentLoaded', () => {
    // Show architecture info once per session
    if (!sessionStorage.getItem('hasSeenModelInfo')) {
        openModal();
        sessionStorage.setItem('hasSeenModelInfo', 'true');
    }
});

function openModal() { document.getElementById('infoModal').style.display = 'flex'; }
function closeModal() { document.getElementById('infoModal').style.display = 'none'; }

// ================= IMAGE HANDLING & API REQUEST =================
function previewImage(event) {
    const file = event.target.files[0];
    if (file) {
        currentFile = file;
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('preview').src = e.target.result;
            document.getElementById('preview').style.display = 'block';
            document.getElementById('upload-text').style.display = 'none';
            // Save base64 to display on the result page
            sessionStorage.setItem('scanImageBase64', e.target.result); 
        }
        reader.readAsDataURL(file);
        logToConsole("File successfully loaded into memory: " + file.name, "#fff");
    }
}

function logToConsole(text, color = "#10b981") {
    const box = document.getElementById('consoleBox');
    box.innerHTML += `<div class="log-line" style="color:${color}">> ${text}</div>`;
    box.scrollTop = box.scrollHeight;
}

async function startDetection() {
    if (!currentFile) { alert("Please upload an image first!"); return; }

    // Simulate forensic terminal output
    document.getElementById('consoleBox').innerHTML = "";
    logToConsole("Initializing OpenCV facial tracking stream...", "#06b6d4");
    setTimeout(() => logToConsole("Isolating Global and Face-crop tensors..."), 400);
    setTimeout(() => logToConsole("Activating Fast Fourier Transform (FFT) analysis..."), 900);
    setTimeout(() => logToConsole("Applying Spatial Rich Model (SRM) noise extraction..."), 1400);
    setTimeout(() => logToConsole("Injecting matrices into B3 Hybrid CNN...", "#f59e0b"), 1900);

    const formData = new FormData();
    formData.append("file", currentFile);

    try {
        // Send request to FastAPI backend
        const response = await fetch("http://localhost:8000/detect", { method: "POST", body: formData });
        const result = await response.json();

        if (!response.ok) throw new Error(result.error);

        setTimeout(() => {
            logToConsole("Verification report generated! Redirecting...", "#fff");
            // Store JSON response for the result page
            sessionStorage.setItem('scanResultData', JSON.stringify(result));
            setTimeout(() => window.location.href = "../result/index.html", 800);
        }, 2500); 

    } catch (error) {
        logToConsole("❌ Backend Error: " + error.message, "#ef4444");
    }
}