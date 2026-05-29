/* ═══════════════════════════════════════════════════════════════
   VESSEL MAP — Circle of Willis SVG
   Interactive anatomical schematic with risk-based coloring
   ═══════════════════════════════════════════════════════════════ */

const VesselMap = (() => {
  // Vessel definitions with SVG path data approximating Circle of Willis anatomy
  const VESSELS = [
    { id: 'l_ic_infra', label: 'L-IC Infra', full: 'Left Infraclinoid ICA', 
      path: 'M 155 340 Q 155 310 150 280 Q 145 255 148 230' },
    { id: 'r_ic_infra', label: 'R-IC Infra', full: 'Right Infraclinoid ICA', 
      path: 'M 245 340 Q 245 310 250 280 Q 255 255 252 230' },
    { id: 'l_ic_supra', label: 'L-IC Supra', full: 'Left Supraclinoid ICA', 
      path: 'M 148 230 Q 145 210 140 195 Q 135 180 138 165' },
    { id: 'r_ic_supra', label: 'R-IC Supra', full: 'Right Supraclinoid ICA', 
      path: 'M 252 230 Q 255 210 260 195 Q 265 180 262 165' },
    { id: 'l_mca', label: 'L-MCA', full: 'Left Middle Cerebral Artery', 
      path: 'M 138 165 Q 125 158 110 152 Q 90 145 70 140' },
    { id: 'r_mca', label: 'R-MCA', full: 'Right Middle Cerebral Artery', 
      path: 'M 262 165 Q 275 158 290 152 Q 310 145 330 140' },
    { id: 'acoma', label: 'AComA', full: 'Anterior Communicating Artery', 
      path: 'M 170 120 L 230 120' },
    { id: 'l_aca', label: 'L-ACA', full: 'Left Anterior Cerebral Artery', 
      path: 'M 138 165 Q 142 150 150 135 Q 158 125 170 120' },
    { id: 'r_aca', label: 'R-ACA', full: 'Right Anterior Cerebral Artery', 
      path: 'M 262 165 Q 258 150 250 135 Q 242 125 230 120' },
    { id: 'l_pcoma', label: 'L-PComA', full: 'Left Posterior Communicating Artery', 
      path: 'M 148 230 Q 155 225 165 222 Q 175 220 185 218' },
    { id: 'r_pcoma', label: 'R-PComA', full: 'Right Posterior Communicating Artery', 
      path: 'M 252 230 Q 245 225 235 222 Q 225 220 215 218' },
    { id: 'basilar', label: 'Basilar', full: 'Basilar Tip', 
      path: 'M 200 300 Q 200 275 200 250 Q 200 235 200 218' },
    { id: 'other_post', label: 'Other Post', full: 'Other Posterior Circulation', 
      path: 'M 185 218 Q 190 215 200 218 Q 210 215 215 218' },
  ];

  // Label positions for each vessel
  const LABEL_POS = {
    l_ic_infra: { x: 110, y: 305 },
    r_ic_infra: { x: 268, y: 305 },
    l_ic_supra: { x: 95, y: 195 },
    r_ic_supra: { x: 275, y: 195 },
    l_mca: { x: 55, y: 132 },
    r_mca: { x: 310, y: 132 },
    acoma: { x: 200, y: 108 },
    l_aca: { x: 135, y: 118 },
    r_aca: { x: 248, y: 118 },
    l_pcoma: { x: 135, y: 240 },
    r_pcoma: { x: 248, y: 240 },
    basilar: { x: 215, y: 270 },
    other_post: { x: 200, y: 205 },
  };

  let currentProbs = {};
  let tooltip = null;

  function getRiskColor(prob) {
    if (prob >= 0.7) return 'var(--status-critical)';
    if (prob >= 0.3) return 'var(--status-elevated)';
    return 'var(--status-clear)';
  }

  function getStrokeWidth(prob) {
    if (prob >= 0.7) return 2.5;
    if (prob >= 0.3) return 2;
    return 1.5;
  }

  function init(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    tooltip = document.getElementById('tooltip');

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 400 380');
    svg.setAttribute('fill', 'none');

    // Background circle (brain outline hint)
    const bgCircle = document.createElementNS('http://www.w3.org/2000/svg', 'ellipse');
    bgCircle.setAttribute('cx', '200');
    bgCircle.setAttribute('cy', '200');
    bgCircle.setAttribute('rx', '170');
    bgCircle.setAttribute('ry', '160');
    bgCircle.setAttribute('stroke', '#E2E8F0');
    bgCircle.setAttribute('stroke-width', '0.5');
    bgCircle.setAttribute('stroke-dasharray', '4 6');
    bgCircle.setAttribute('fill', 'none');
    svg.appendChild(bgCircle);

    // Draw vessels
    VESSELS.forEach(v => {
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', v.path);
      path.setAttribute('stroke', '#CBD5E1');
      path.setAttribute('stroke-width', '1.5');
      path.setAttribute('stroke-linecap', 'round');
      path.setAttribute('fill', 'none');
      path.setAttribute('id', `vessel-${v.id}`);
      path.setAttribute('data-vessel', v.id);
      path.style.transition = 'stroke 0.4s ease, stroke-width 0.3s ease';
      path.style.cursor = 'pointer';

      path.addEventListener('mouseenter', (e) => showTooltip(e, v));
      path.addEventListener('mouseleave', hideTooltip);
      path.addEventListener('mousemove', moveTooltip);

      svg.appendChild(path);
    });

    // Draw labels
    VESSELS.forEach(v => {
      const pos = LABEL_POS[v.id];
      if (!pos) return;
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', pos.x);
      text.setAttribute('y', pos.y);
      text.setAttribute('fill', '#64748B');
      text.setAttribute('font-family', "'Inter', sans-serif");
      text.setAttribute('font-size', '8');
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('id', `label-${v.id}`);
      text.textContent = v.label;
      svg.appendChild(text);
    });

    container.appendChild(svg);
  }

  function showTooltip(e, vessel) {
    if (!tooltip) return;
    const prob = currentProbs[vessel.id];
    const pStr = prob !== undefined ? (prob * 100).toFixed(1) + '%' : '—';
    tooltip.textContent = `${vessel.full}: ${pStr}`;
    tooltip.classList.add('visible');
    moveTooltip(e);
  }

  function moveTooltip(e) {
    if (!tooltip) return;
    tooltip.style.left = (e.clientX + 12) + 'px';
    tooltip.style.top = (e.clientY - 8) + 'px';
  }

  function hideTooltip() {
    if (!tooltip) return;
    tooltip.classList.remove('visible');
  }

  function update(probabilities) {
    // probabilities: { l_ic_infra: 0.02, ... }
    currentProbs = probabilities;

    VESSELS.forEach(v => {
      const path = document.getElementById(`vessel-${v.id}`);
      const label = document.getElementById(`label-${v.id}`);
      const prob = probabilities[v.id] || 0;

      if (path) {
        path.setAttribute('stroke', getRiskColor(prob));
        path.setAttribute('stroke-width', String(getStrokeWidth(prob)));
        
        if (prob >= 0.7) {
          path.classList.add('pulse-critical');
        } else {
          path.classList.remove('pulse-critical');
        }
      }

      if (label) {
        label.setAttribute('fill', prob >= 0.3 ? '#0F172A' : '#64748B');
      }
    });
  }

  return { init, update, VESSELS };
})();

// Expose globally
window.VesselMap = VesselMap;
