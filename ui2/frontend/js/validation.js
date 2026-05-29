/* ═══════════════════════════════════════════════════════════════
   CLINICAL VALIDATION — Metric Animations & Scroll Reveal
   Handles counting up validation statistics, and manages viewport
   reveal animations for high-precision components.
   ═══════════════════════════════════════════════════════════════ */

const ClinicalValidation = (() => {
  let revealElements = [];
  let animatedMetrics = new Set();

  function init() {
    revealElements = Array.from(document.querySelectorAll('.reveal'));
    
    // Set up Intersection Observer for fade-in slide-up reveals
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          
          // If the entry contains a metric card, trigger count up
          if (entry.target.classList.contains('metric-card')) {
            const valueEl = entry.target.querySelector('.metric-card-value');
            if (valueEl && !animatedMetrics.has(valueEl.id)) {
              animateMetric(valueEl);
            }
          }
        }
      });
    }, {
      threshold: 0.1,
      rootMargin: "0px 0px -50px 0px"
    });

    revealElements.forEach(el => {
      observer.observe(el);
    });
  }

  function animateMetric(el) {
    animatedMetrics.add(el.id);
    const targetVal = parseFloat(el.getAttribute('data-target'));
    const suffix = el.getAttribute('data-suffix') || "";
    const isFloat = targetVal < 1.0;
    
    let startVal = 0.0;
    const duration = 1200; // ms
    const startTime = performance.now();

    function update(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1.0);
      
      // Easing function (easeOutQuad)
      const easeProgress = progress * (2 - progress);
      const currentVal = easeProgress * targetVal;
      
      if (isFloat) {
        el.textContent = currentVal.toFixed(3) + suffix;
      } else {
        el.textContent = currentVal.toFixed(1) + suffix;
      }

      if (progress < 1.0) {
        requestAnimationFrame(update);
      } else {
        // Guarantee exact target value at the end
        if (isFloat) {
          el.textContent = targetVal.toFixed(3) + suffix;
        } else {
          el.textContent = targetVal.toFixed(1) + suffix;
        }
      }
    }

    requestAnimationFrame(update);
  }

  // Initialize once loaded
  document.addEventListener('DOMContentLoaded', init);

  return {
    init
  };
})();

// Expose globally
window.ClinicalValidation = ClinicalValidation;
