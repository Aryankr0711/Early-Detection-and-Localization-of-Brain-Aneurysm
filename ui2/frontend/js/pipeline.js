/* ═══════════════════════════════════════════════════════════════
   PIPELINE — Neural Architecture Stage Visualizer
   Handles active stage highlighting and integration with the
   inference state machine.
   ═══════════════════════════════════════════════════════════════ */

const Pipeline = (() => {
  let stages = [];

  function init() {
    stages = Array.from(document.querySelectorAll('.pipeline-stage'));
    const scrollContainer = document.querySelector('.pipeline-scroll-container');
    
    if (scrollContainer) {
      window.addEventListener('scroll', () => {
        // If inference is running, do not let scroll override the pipeline state
        if (document.getElementById('processingContainer')?.classList.contains('active')) return;

        const rect = scrollContainer.getBoundingClientRect();
        // Distance the user scrolls while the container is sticky
        const scrollRange = rect.height - window.innerHeight;
        
        // Progress from 0 (top of container hits top of viewport) to 1 (bottom of container leaves)
        let progress = -rect.top / scrollRange;
        progress = Math.max(0, Math.min(1, progress));
        
        // Reset all
        stages.forEach(s => s.classList.remove('active'));

        // Staggered architectural reveal
        if (progress > 0.1) stages[0].classList.add('active');
        if (progress > 0.4) stages[1].classList.add('active');
        if (progress > 0.7) stages[2].classList.add('active');
      });
    }
  }

  /**
   * Sets the active stage in the pipeline (1, 2, or 3).
   * Pass 0 to deactivate all stages.
   * @param {number} stageNum 
   */
  function setStage(stageNum) {
    stages.forEach((stage, idx) => {
      // In inference mode, we light up stages progressively
      if ((idx + 1) <= stageNum) {
        stage.classList.add('active');
      } else {
        stage.classList.remove('active');
      }
    });
  }

  // Initialize once loaded
  document.addEventListener('DOMContentLoaded', init);

  return {
    setStage
  };
})();

// Expose globally
window.Pipeline = Pipeline;
