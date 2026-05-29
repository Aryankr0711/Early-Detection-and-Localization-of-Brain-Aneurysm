/* ═══════════════════════════════════════════════════════════════
   DICOM VIEWER — Multi-Planar Reconstruction (MPR)
   Procedural radiology-grade slice renderer for Axial, Coronal,
   and Sagittal viewports with dynamic contrast/windowing, crosshairs,
   and localized aneurysm annotations.
   ═══════════════════════════════════════════════════════════════ */

const DicomViewer = (() => {
  let canvases = { axial: null, coronal: null, sagittal: null };
  let ctxs = { axial: null, coronal: null, sagittal: null };
  let scrubber = null;
  let scrubberLabel = null;
  
  let currentSlice = 93;
  let maxSlices = 186;
  let activeCase = null;
  let activeVesselKey = null; // Currently highlighted vessel from results table
  
  // Windowing settings
  let windowWidth = 400;
  let windowLevel = 40;
  let isDraggingWL = false;
  let dragStart = { x: 0, y: 0 };
  let dragStartWL = { w: 400, l: 40 };

  function init() {
    canvases.axial = document.getElementById('canvasAxial');
    canvases.coronal = document.getElementById('canvasCoronal');
    canvases.sagittal = document.getElementById('canvasSagittal');
    
    if (!canvases.axial || !canvases.coronal || !canvases.sagittal) {
      console.log("DICOM Viewer elements not found. Viewer disabled.");
      return;
    }
    
    ctxs.axial = canvases.axial.getContext('2d');
    ctxs.coronal = canvases.coronal.getContext('2d');
    ctxs.sagittal = canvases.sagittal.getContext('2d');
    
    scrubber = document.getElementById('sliceScrubber');
    scrubberLabel = document.getElementById('scrubberLabel');

    // Handle range slider input
    if (scrubber) {
      scrubber.addEventListener('input', (e) => {
        setSlice(parseInt(e.target.value, 10));
      });
    }

    // Handle windowing adjustment via dragging
    Object.keys(canvases).forEach(key => {
      const canvas = canvases[key];
      if (canvas) {
        canvas.addEventListener('mousedown', (e) => {
          isDraggingWL = true;
          dragStart.x = e.clientX;
          dragStart.y = e.clientY;
          dragStartWL.w = windowWidth;
          dragStartWL.l = windowLevel;
          e.preventDefault();
        });
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDraggingWL) return;
      const dx = e.clientX - dragStart.x;
      const dy = e.clientY - dragStart.y;
      
      // Horizontal drag controls Window Width, vertical controls Window Level
      windowWidth = Math.max(50, Math.min(1000, dragStartWL.w + dx * 2));
      windowLevel = Math.max(-200, Math.min(400, dragStartWL.l - dy));
      
      updateWLDisplays();
      drawAll();
    });

    window.addEventListener('mouseup', () => {
      isDraggingWL = false;
    });

    // Set up canvas sizes on resize
    window.addEventListener('resize', resizeCanvases);
    resizeCanvases();
  }

  function resizeCanvases() {
    if (!canvases.axial) return;
    Object.keys(canvases).forEach(key => {
      const canvas = canvases[key];
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      // Account for device pixel ratio for crisp rendering
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.width * dpr; // Square viewports
    });
    drawAll();
  }

  function setSlice(sliceIndex) {
    if (!canvases.axial) return;
    currentSlice = Math.max(0, Math.min(maxSlices - 1, sliceIndex));
    if (scrubber) scrubber.value = currentSlice;
    if (scrubberLabel) scrubberLabel.textContent = `Slice: ${currentSlice + 1}/${maxSlices}`;
    
    // Update individual viewport metadata
    const metaAxial = document.getElementById('metaAxial');
    if (metaAxial) metaAxial.textContent = `Slice: ${currentSlice + 1}/${maxSlices}`;
    
    const metaCoronal = document.getElementById('metaCoronal');
    if (metaCoronal) metaCoronal.textContent = `Y-Plane: ${Math.floor(currentSlice * 1.2)}/220`;
    
    const metaSagittal = document.getElementById('metaSagittal');
    if (metaSagittal) metaSagittal.textContent = `X-Plane: ${Math.floor(currentSlice * 1.2)}/220`;
    
    drawAll();
  }

  function updateWLDisplays() {
    const wlTexts = document.querySelectorAll('.dicom-viewport-wl');
    wlTexts.forEach(el => {
      el.textContent = `W:${windowWidth} L:${windowLevel}`;
    });
  }

  function setCase(caseData) {
    if (!canvases.axial) return;
    maxSlices = parseInt(caseData.telemetry.slices, 10);
    if (scrubber) {
      scrubber.max = maxSlices - 1;
    }
    // Jump to the primary critical slice if aneurysm exists, else center slice
    let targetSlice = Math.floor(maxSlices / 2);
    if (caseData.vessels) {
      const criticalVessels = Object.keys(caseData.vessels).filter(k => caseData.vessels[k].risk === "CRITICAL");
      if (criticalVessels.length > 0) {
        targetSlice = caseData.vessels[criticalVessels[0]].centerSlice;
      }
    }
    setSlice(targetSlice);
  }

  function selectVessel(vesselKey) {
    if (!canvases.axial) return;
    activeVesselKey = vesselKey;
    if (activeCase && activeCase.vessels[vesselKey]) {
      const vesselInfo = activeCase.vessels[vesselKey];
      if (vesselInfo.centerSlice > 0) {
        setSlice(vesselInfo.centerSlice);
      }
    }
    drawAll();
  }

  function drawAll() {
    if (!canvases.axial || !canvases.axial.width) return;
    drawAxial();
    drawCoronal();
    drawSagittal();
  }

  // Helper to get brightness multiplier based on WL values
  // Higher level = darker; smaller width = higher contrast
  function getWLEffects() {
    const contrast = 400 / windowWidth; // base normal width is 400
    const brightness = (40 - windowLevel) / 200; // base normal level is 40
    return { contrast, brightness };
  }

  /**
   * AXIAL VIEWPORT (Horizontal slice through Z)
   */
  function drawAxial() {
    const canvas = canvases.axial;
    const ctx = ctxs.axial;
    const w = canvas.width;
    const h = canvas.height;
    const wl = getWLEffects();
    
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, w, h);
    
    // Scale factor depending on slice (simulating brain shape changing along Z-axis)
    // 0 = neck/base, 0.5 = middle (widest), 1.0 = top of skull
    const normSlice = currentSlice / maxSlices;
    const brainScale = Math.sin(normSlice * Math.PI) * 0.85 + 0.15;
    
    if (brainScale <= 0.2) {
      // Draw outer skull bone contour (very small/empty)
      return;
    }
    
    ctx.save();
    ctx.translate(w/2, h/2);
    ctx.scale(brainScale, brainScale);
    
    // Apply W/L contrast and brightness
    ctx.filter = `contrast(${wl.contrast}) brightness(${1 + wl.brightness})`;
    
    // 1. Draw Skull Bone (thick grey boundary)
    ctx.strokeStyle = '#333338';
    ctx.lineWidth = 14;
    ctx.beginPath();
    ctx.ellipse(0, 0, w * 0.38, h * 0.44, 0, 0, 2 * Math.PI);
    ctx.stroke();
    
    ctx.strokeStyle = '#18181A';
    ctx.lineWidth = 2;
    ctx.stroke();

    // 2. Draw Brain Tissue (light grey filled shape)
    ctx.fillStyle = '#1A1D24';
    ctx.beginPath();
    ctx.ellipse(0, 0, w * 0.35, h * 0.41, 0, 0, 2 * Math.PI);
    ctx.fill();
    
    // Ventricles (dark cavities inside brain)
    ctx.fillStyle = '#05070A';
    ctx.beginPath();
    // Left Lateral Ventricle (butterfly wing)
    ctx.moveTo(-10, -30);
    ctx.bezierCurveTo(-25, -50, -40, -10, -5, 20);
    ctx.bezierCurveTo(-15, 0, -15, -20, -10, -30);
    // Right Lateral Ventricle
    ctx.moveTo(10, -30);
    ctx.bezierCurveTo(25, -50, 40, -10, 5, 20);
    ctx.bezierCurveTo(15, 0, 15, -20, 10, -30);
    ctx.fill();
    
    // 3. Draw Vasculature (contrast CTA - bright vessel lines)
    ctx.filter = 'none'; // reset filter for bright overlay vessels
    ctx.strokeStyle = 'rgba(240, 245, 255, 0.45)';
    ctx.lineWidth = 1.5;
    
    // Draw Circle of Willis branches
    ctx.beginPath();
    // Basilar tip -> PCA split
    ctx.moveTo(0, 50);
    ctx.lineTo(0, 15);
    ctx.lineTo(-25, 5); // Left PCA
    ctx.moveTo(0, 15);
    ctx.lineTo(25, 5);  // Right PCA
    
    // ICA -> MCA split
    ctx.moveTo(-35, -5);
    ctx.lineTo(-75, -15); // Left MCA
    ctx.moveTo(35, -5);
    ctx.lineTo(75, -15);  // Right MCA
    
    // ICA -> ACA
    ctx.moveTo(-35, -5);
    ctx.lineTo(-15, -45); // Left ACA
    ctx.moveTo(35, -5);
    ctx.lineTo(15, -45);  // Right ACA
    
    // AComA bridging ACA
    ctx.moveTo(-15, -45);
    ctx.lineTo(15, -45);
    ctx.stroke();
    
    // Ambient breathing pulse on vessels near finding
    const isCriticalCase = activeCase && activeCase.apValue > 0.5;
    
    // 4. Draw Aneurysm Annotation if close to the slice
    if (activeCase && isCriticalCase) {
      Object.keys(activeCase.vessels).forEach(vesselKey => {
        const v = activeCase.vessels[vesselKey];
        if (v.centerSlice > 0 && Math.abs(currentSlice - v.centerSlice) < 6) {
          // Calculate opacity based on slice distance
          const sliceDist = Math.abs(currentSlice - v.centerSlice);
          const opacity = 1 - (sliceDist / 6);
          const isSelected = activeVesselKey === vesselKey;
          
          // Draw the aneurysm lesion (bright glowing sphere)
          // Map anatomical location to axial canvas coords (procedural mapping)
          let cx = 0;
          let cy = 0;
          
          if (vesselKey === "AComA") { cx = 0; cy = -45; }
          else if (vesselKey === "L-MCA") { cx = -50; cy = -10; }
          else if (vesselKey === "R-MCA") { cx = 50; cy = -10; }
          
          if (cx !== 0 || cy !== 0) {
            // Draw localized circle
            ctx.fillStyle = v.risk === "CRITICAL" ? `rgba(239, 68, 68, ${opacity * 0.85})` : `rgba(245, 158, 11, ${opacity * 0.85})`;
            ctx.beginPath();
            ctx.arc(cx, cy, v.radius * (brainScale * 0.9), 0, 2 * Math.PI);
            ctx.fill();
            
            // Outer bright border
            ctx.strokeStyle = v.risk === "CRITICAL" ? `rgba(239, 68, 68, ${opacity})` : `rgba(245, 158, 11, ${opacity})`;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(cx, cy, (v.radius + 3) * (brainScale * 0.9), 0, 2 * Math.PI);
            ctx.stroke();
            
            // Crosshairs if this vessel is selected or active
            if (isSelected && sliceDist === 0) {
              ctx.strokeStyle = 'rgba(0, 180, 216, 0.7)';
              ctx.lineWidth = 1;
              ctx.setLineDash([3, 3]);
              
              // Horizontal crosshair
              ctx.beginPath();
              ctx.moveTo(-w/2, cy);
              ctx.lineTo(w/2, cy);
              ctx.stroke();
              
              // Vertical crosshair
              ctx.beginPath();
              ctx.moveTo(cx, -h/2);
              ctx.lineTo(cx, h/2);
              ctx.stroke();
              
              ctx.setLineDash([]);
              
              // Label
              ctx.fillStyle = 'var(--accent-cyan)';
              ctx.font = `bold ${Math.max(9, 10 / brainScale)}px var(--font-data)`;
              ctx.fillText(`${vesselKey}: ${(v.prob * 100).toFixed(1)}%`, cx + v.radius + 8, cy + 3);
            }
          }
        }
      });
    }
    
    ctx.restore();
  }

  /**
   * CORONAL VIEWPORT (Vertical front-to-back slice)
   */
  function drawCoronal() {
    const canvas = canvases.coronal;
    const ctx = ctxs.coronal;
    const w = canvas.width;
    const h = canvas.height;
    const wl = getWLEffects();
    
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, w, h);
    
    // Scale shape
    const normSlice = currentSlice / maxSlices;
    const scale = Math.sin(normSlice * Math.PI) * 0.8 + 0.2;
    
    ctx.save();
    ctx.translate(w/2, h/2);
    ctx.scale(scale, scale);
    
    ctx.filter = `contrast(${wl.contrast}) brightness(${1 + wl.brightness})`;
    
    // Skull
    ctx.strokeStyle = '#333338';
    ctx.lineWidth = 12;
    ctx.beginPath();
    ctx.ellipse(0, -10, w * 0.36, h * 0.40, 0, 0, 2 * Math.PI);
    ctx.stroke();
    
    // Brain Tissue
    ctx.fillStyle = '#1A1D24';
    ctx.beginPath();
    ctx.ellipse(0, -10, w * 0.33, h * 0.37, 0, 0, 2 * Math.PI);
    ctx.fill();
    
    // Midbrain ventricles (vertical split line in center coronal)
    ctx.fillStyle = '#05070A';
    ctx.beginPath();
    ctx.ellipse(0, -15, w * 0.04, h * 0.16, 0, 0, 2 * Math.PI);
    ctx.fill();
    
    // Vasculature (Coronal projections - PCA/MCA ascending branches)
    ctx.filter = 'none';
    ctx.strokeStyle = 'rgba(240, 245, 255, 0.4)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    // Basilar artery rising
    ctx.moveTo(0, 80);
    ctx.lineTo(0, 10);
    // Splitting into PCA/MCA
    ctx.lineTo(-40, -15);
    ctx.moveTo(0, 10);
    ctx.lineTo(40, -15);
    ctx.stroke();
    
    // Draw Aneurysm lesion if active
    if (activeCase && activeCase.apValue > 0.5) {
      Object.keys(activeCase.vessels).forEach(vesselKey => {
        const v = activeCase.vessels[vesselKey];
        if (v.centerSlice > 0 && Math.abs(currentSlice - v.centerSlice) < 6) {
          const sliceDist = Math.abs(currentSlice - v.centerSlice);
          const opacity = 1 - (sliceDist / 6);
          const isSelected = activeVesselKey === vesselKey;
          
          let cx = 0;
          let cy = 0;
          if (vesselKey === "AComA") { cx = 0; cy = -20; }
          else if (vesselKey === "L-MCA") { cx = -35; cy = -12; }
          else if (vesselKey === "R-MCA") { cx = 35; cy = -12; }
          
          if (cx !== 0 || cy !== 0) {
            ctx.fillStyle = v.risk === "CRITICAL" ? `rgba(239, 68, 68, ${opacity * 0.8})` : `rgba(245, 158, 11, ${opacity * 0.8})`;
            ctx.beginPath();
            ctx.arc(cx, cy, v.radius * scale, 0, 2 * Math.PI);
            ctx.fill();
            
            if (isSelected && sliceDist === 0) {
              ctx.strokeStyle = 'rgba(0, 180, 216, 0.6)';
              ctx.lineWidth = 1;
              ctx.setLineDash([3, 3]);
              ctx.beginPath();
              ctx.moveTo(-w/2, cy); ctx.lineTo(w/2, cy);
              ctx.moveTo(cx, -h/2); ctx.lineTo(cx, h/2);
              ctx.stroke();
              ctx.setLineDash([]);
            }
          }
        }
      });
    }
    
    ctx.restore();
  }

  /**
   * SAGITTAL VIEWPORT (Side profile slice)
   */
  function drawSagittal() {
    const canvas = canvases.sagittal;
    const ctx = ctxs.sagittal;
    const w = canvas.width;
    const h = canvas.height;
    const wl = getWLEffects();
    
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, w, h);
    
    const normSlice = currentSlice / maxSlices;
    const scale = Math.sin(normSlice * Math.PI) * 0.78 + 0.22;
    
    ctx.save();
    ctx.translate(w/2, h/2);
    ctx.scale(scale, scale);
    
    ctx.filter = `contrast(${wl.contrast}) brightness(${1 + wl.brightness})`;
    
    // Skull (egg-like profile)
    ctx.strokeStyle = '#333338';
    ctx.lineWidth = 12;
    ctx.beginPath();
    ctx.ellipse(-10, -15, w * 0.42, h * 0.35, 0.1, 0, 2 * Math.PI);
    ctx.stroke();
    
    // Brain
    ctx.fillStyle = '#1A1D24';
    ctx.beginPath();
    ctx.ellipse(-10, -15, w * 0.39, h * 0.32, 0.1, 0, 2 * Math.PI);
    ctx.fill();
    
    // Cerebellum (mini-brain shape at posterior bottom)
    ctx.fillStyle = '#14171D';
    ctx.beginPath();
    ctx.ellipse(-45, 45, w * 0.16, h * 0.12, 0, 0, 2 * Math.PI);
    ctx.fill();
    
    // Brainstem (connecting descending trunk)
    ctx.fillStyle = '#181B22';
    ctx.beginPath();
    ctx.ellipse(-5, 65, w * 0.08, h * 0.20, 0.2, 0, 2 * Math.PI);
    ctx.fill();
    
    // Vasculature (Carinal/Sagittal profile - loop of carotid/vertebral)
    ctx.filter = 'none';
    ctx.strokeStyle = 'rgba(240, 245, 255, 0.4)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    // Vertebral/Basilar curve
    ctx.moveTo(-15, 80);
    ctx.bezierCurveTo(-12, 40, -2, 20, 10, 0);
    // Anterior carotid curve (siphon)
    ctx.moveTo(35, 70);
    ctx.bezierCurveTo(30, 25, 20, 5, 12, -8);
    ctx.stroke();
    
    // Draw Aneurysm lesion if active
    if (activeCase && activeCase.apValue > 0.5) {
      Object.keys(activeCase.vessels).forEach(vesselKey => {
        const v = activeCase.vessels[vesselKey];
        if (v.centerSlice > 0 && Math.abs(currentSlice - v.centerSlice) < 6) {
          const sliceDist = Math.abs(currentSlice - v.centerSlice);
          const opacity = 1 - (sliceDist / 6);
          const isSelected = activeVesselKey === vesselKey;
          
          let cx = 0;
          let cy = 0;
          if (vesselKey === "AComA") { cx = 16; cy = -12; }
          else if (vesselKey === "L-MCA") { cx = 10; cy = 2; }
          else if (vesselKey === "R-MCA") { cx = 10; cy = 2; }
          
          if (cx !== 0 || cy !== 0) {
            ctx.fillStyle = v.risk === "CRITICAL" ? `rgba(239, 68, 68, ${opacity * 0.8})` : `rgba(245, 158, 11, ${opacity * 0.8})`;
            ctx.beginPath();
            ctx.arc(cx, cy, v.radius * scale, 0, 2 * Math.PI);
            ctx.fill();
            
            if (isSelected && sliceDist === 0) {
              ctx.strokeStyle = 'rgba(0, 180, 216, 0.6)';
              ctx.lineWidth = 1;
              ctx.setLineDash([3, 3]);
              ctx.beginPath();
              ctx.moveTo(-w/2, cy); ctx.lineTo(w/2, cy);
              ctx.moveTo(cx, -h/2); ctx.lineTo(cx, h/2);
              ctx.stroke();
              ctx.setLineDash([]);
            }
          }
        }
      });
    }
    
    ctx.restore();
  }

  // Initialize once DOM is ready
  document.addEventListener('DOMContentLoaded', init);

  return {
    setCase,
    setSlice,
    selectVessel,
    drawAll
  };
})();

// Expose globally
window.DicomViewer = DicomViewer;
