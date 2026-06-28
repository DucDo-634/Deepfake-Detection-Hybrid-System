document.addEventListener("DOMContentLoaded", () => {
    // 1. Retrieve data from session storage
    const resultDataRaw = sessionStorage.getItem('scanResultData');
    const imageBase64 = sessionStorage.getItem('scanImageBase64');

    if (!resultDataRaw || !imageBase64) {
        alert("No analysis data found. Please run a scan first!");
        window.location.href = "../detect/index.html";
        return;
    }

    const data = JSON.parse(resultDataRaw);

    // 2. Populate Image and Status
    document.getElementById('res-image').src = imageBase64;
    
    const statusDiv = document.getElementById('res-status');
    statusDiv.innerText = data.status;
    
    // 3. UI Color Routing based on 3-tier Logic
    const progressBar = document.getElementById('res-progress-bar');
    statusDiv.className = "res-status"; 
    
    if (data.status.includes("AUTHENTIC")) {
        statusDiv.classList.add('status-real');
        progressBar.style.backgroundColor = "var(--success)"; 
    } else if (data.status.includes("SUSPICIOUS")) {
        statusDiv.classList.add('status-suspicious');
        progressBar.style.backgroundColor = "var(--warning)"; 
    } else {
        statusDiv.classList.add('status-fake');
        progressBar.style.backgroundColor = "var(--danger)"; 
    }

    // 4. Inject Metrics
    document.getElementById('res-comment').innerText = data.metrics.evaluation;
    document.getElementById('res-trigger').innerText = data.metrics.trigger_stream;
    document.getElementById('res-logit').innerText = data.metrics.final_ai_score.toFixed(4);
    
    // Handle Progress Bar Animation
    const confString = data.confidence;
    document.getElementById('res-conf').innerText = confString;
    
    setTimeout(() => {
        progressBar.style.width = confString;
    }, 300);
});