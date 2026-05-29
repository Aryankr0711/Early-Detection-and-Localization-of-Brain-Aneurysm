/* ═══════════════════════════════════════════════════════════════
   APPLICATION CORE — SPA Router & Navigation Controller
   Manages navbar scroll-spy, toggles Results/Viewer navigation
   link visibility, and boots the entire workstation interface.
   ═══════════════════════════════════════════════════════════════ */

const App = (() => {
  let navLinks = [];
  let sections = [];
  let resultsNavLink = null;
  let viewerNavLink = null;

  function init() {
    navLinks = Array.from(document.querySelectorAll('.nav-links a'));
    sections = Array.from(document.querySelectorAll('section'));
    
    resultsNavLink = document.querySelector('a[data-nav="results"]').parentElement;
    const viewerLinkEl = document.querySelector('a[data-nav="viewer"]');
    viewerNavLink = viewerLinkEl ? viewerLinkEl.parentElement : null;

    // Initially hide results and imaging from the nav until a case is run
    hideResultsNav();

    // 1. Navigation Click Event Handlers
    navLinks.forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href');
        const targetSection = document.querySelector(targetId);
        
        if (targetSection) {
          // If section is currently display: none (like results or viewer prior to upload),
          // don't try to scroll.
          if (getComputedStyle(targetSection).display === 'none') {
            return;
          }

          targetSection.scrollIntoView({
            behavior: 'smooth'
          });

          // Set active navbar element
          setActiveNavLink(link.getAttribute('data-nav'));
        }
      });
    });

    // 2. Scroll Spy (Intersection Observer to auto-update active nav links)
    const scrollObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const sectionId = entry.target.getAttribute('id');
          // Only update link state if the section is visible
          if (getComputedStyle(entry.target).display !== 'none') {
            setActiveNavLink(sectionId);
          }
        }
      });
    }, {
      threshold: 0.3,
      rootMargin: "-48px 0px -50% 0px" // Account for navbar height
    });

    sections.forEach(section => {
      scrollObserver.observe(section);
    });

    // 3. Bind events for displaying the results tabs
    const originalLoadResults = Results.loadResults;
    Results.loadResults = function(caseData) {
      showResultsNav();
      originalLoadResults(caseData);
    };
  }

  function hideResultsNav() {
    if (resultsNavLink) resultsNavLink.style.display = 'none';
    if (viewerNavLink) viewerNavLink.style.display = 'none';
  }

  function showResultsNav() {
    if (resultsNavLink) resultsNavLink.style.display = 'block';
    if (viewerNavLink) viewerNavLink.style.display = 'block';
  }

  function setActiveNavLink(navKey) {
    navLinks.forEach(link => {
      if (link.getAttribute('data-nav') === navKey) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });
  }

  // Initialize
  document.addEventListener('DOMContentLoaded', init);

  return {
    hideResultsNav,
    showResultsNav
  };
})();

// Expose globally
window.App = App;
