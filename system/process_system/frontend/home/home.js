// ================= INTERSECTION OBSERVER FOR 3D SCROLL ANIMATION =================
document.addEventListener('DOMContentLoaded', () => {
    const items = document.querySelectorAll('.scroll-item');
    
    // Trigger animation when 15% of the card is visible in the viewport
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('show');
            } else {
                // Remove class when scrolling up to allow re-animation
                entry.target.classList.remove('show');
            }
        });
    }, { threshold: 0.15, rootMargin: "0px 0px -50px 0px" });

    items.forEach(item => observer.observe(item));
});