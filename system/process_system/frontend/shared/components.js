

const headerHTML = `
<header>
    <a href="../home/index.html" class="logo">
        AI<span>Forensics</span>
    </a>
    <nav>
        <a href="../home/index.html" id="nav-home">Dashboard</a>
        <a href="../detect/index.html" id="nav-detect">Analysis Engine</a>
    </nav>
</header>
`;

const footerHTML = `
<footer>
    <p>&copy; 2026 <span>Deepfake Forensics</span>. Dual-stream Architecture Engine.</p>
</footer>
`;

document.addEventListener("DOMContentLoaded", () => {
    // Inject Header and Footer
    document.getElementById("header-placeholder").innerHTML = headerHTML;
    document.getElementById("footer-placeholder").innerHTML = footerHTML;

    // Auto-highlight active navigation menu
    const path = window.location.pathname;
    if (path.includes("home")) document.getElementById("nav-home").classList.add("active");
    if (path.includes("detect") || path.includes("result")) document.getElementById("nav-detect").classList.add("active");
});