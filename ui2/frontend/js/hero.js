/* ═══════════════════════════════════════════════════════════════
   HERO — Procedural Vascular-Field Visualization
   Sparse vessel trajectories with ambient neural flow
   ═══════════════════════════════════════════════════════════════ */

const Hero = (() => {
  let canvas, ctx;
  let width, height;
  let branches = [];
  let flowParticles = [];
  let time = 0;
  let animId;

  // Generate branching vascular tree structure
  function generateBranch(x, y, angle, depth, maxDepth, length) {
    if (depth > maxDepth) return;
    
    const segments = [];
    const numPoints = 8 + Math.floor(Math.random() * 6);
    let cx = x, cy = y;
    let ca = angle;
    
    for (let i = 0; i < numPoints; i++) {
      const t = i / numPoints;
      const segLen = length / numPoints;
      ca += (Math.random() - 0.5) * 0.3;
      cx += Math.cos(ca) * segLen;
      cy += Math.sin(ca) * segLen;
      segments.push({ x: cx, y: cy });
    }

    const opacity = 0.04 + (1 - depth / maxDepth) * 0.08;
    const lineWidth = 0.5 + (1 - depth / maxDepth) * 1.2;

    branches.push({ segments, opacity, lineWidth, depth });

    // Branching: deeper levels branch more
    if (depth < maxDepth) {
      const branchCount = depth < 2 ? 2 : (Math.random() > 0.4 ? 2 : 1);
      for (let b = 0; b < branchCount; b++) {
        const branchPoint = segments[Math.floor(segments.length * (0.5 + Math.random() * 0.4))];
        const branchAngle = ca + (Math.random() - 0.5) * 1.2;
        const branchLen = length * (0.5 + Math.random() * 0.25);
        generateBranch(branchPoint.x, branchPoint.y, branchAngle, depth + 1, maxDepth, branchLen);
      }
    }
  }

  function generateVascularField() {
    branches = [];
    
    // Main trunks from bottom-right region, branching upward
    const origins = [
      { x: width * 0.65, y: height * 0.9, angle: -Math.PI * 0.45, len: height * 0.5 },
      { x: width * 0.72, y: height * 0.85, angle: -Math.PI * 0.5, len: height * 0.45 },
      { x: width * 0.8, y: height * 0.95, angle: -Math.PI * 0.55, len: height * 0.55 },
      { x: width * 0.58, y: height * 0.88, angle: -Math.PI * 0.38, len: height * 0.42 },
      { x: width * 0.85, y: height * 0.7, angle: -Math.PI * 0.6, len: height * 0.35 },
      { x: width * 0.5, y: height * 0.95, angle: -Math.PI * 0.35, len: height * 0.4 },
    ];

    origins.forEach(o => {
      generateBranch(o.x, o.y, o.angle, 0, 4, o.len);
    });

    // Generate flow particles along some branches
    flowParticles = [];
    const flowBranches = branches.filter(b => b.segments.length > 4 && b.depth < 3);
    const selectedBranches = flowBranches.sort(() => Math.random() - 0.5).slice(0, 12);
    
    selectedBranches.forEach(branch => {
      flowParticles.push({
        branch,
        t: Math.random(),
        speed: 0.001 + Math.random() * 0.002,
        size: 1 + Math.random() * 1.5,
        opacity: 0.15 + Math.random() * 0.2,
      });
    });
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);
    
    // Ambient breathing: entire field oscillates opacity
    const breathe = 0.85 + Math.sin(time * 0.08) * 0.15;

    // Draw branches
    branches.forEach(branch => {
      if (branch.segments.length < 2) return;
      ctx.beginPath();
      ctx.moveTo(branch.segments[0].x, branch.segments[0].y);
      
      for (let i = 1; i < branch.segments.length; i++) {
        const prev = branch.segments[i - 1];
        const curr = branch.segments[i];
        const cpx = (prev.x + curr.x) / 2;
        const cpy = (prev.y + curr.y) / 2;
        ctx.quadraticCurveTo(prev.x, prev.y, cpx, cpy);
      }
      
      ctx.strokeStyle = `rgba(15, 23, 42, ${branch.opacity * 5.5 * breathe})`;
      ctx.lineWidth = branch.lineWidth;
      ctx.lineCap = 'round';
      ctx.stroke();
    });

    // Draw flow particles
    flowParticles.forEach(p => {
      p.t += p.speed;
      if (p.t > 1) p.t = 0;

      const segments = p.branch.segments;
      const idx = Math.floor(p.t * (segments.length - 1));
      const nextIdx = Math.min(idx + 1, segments.length - 1);
      const localT = (p.t * (segments.length - 1)) - idx;
      
      const x = segments[idx].x + (segments[nextIdx].x - segments[idx].x) * localT;
      const y = segments[idx].y + (segments[nextIdx].y - segments[idx].y) * localT;

      const grad = ctx.createRadialGradient(x, y, 0, x, y, p.size * 3.5);
      grad.addColorStop(0, `rgba(29, 78, 216, ${p.opacity * 3.5 * breathe})`);
      grad.addColorStop(1, 'rgba(29, 78, 216, 0)');

      ctx.beginPath();
      ctx.arc(x, y, p.size * 3.5, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();

      // Core dot
      ctx.beginPath();
      ctx.arc(x, y, p.size * 1.2, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(29, 78, 216, ${(p.opacity * 2.5 + 0.3) * breathe})`;
      ctx.fill();
    });

    time++;
    animId = requestAnimationFrame(draw);
  }

  function init() {
    canvas = document.getElementById('heroCanvas');
    if (!canvas) return;
    ctx = canvas.getContext('2d');
    resize();
    generateVascularField();
    draw();
    window.addEventListener('resize', resize);
  }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    width = rect.width;
    height = rect.height;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.scale(dpr, dpr);
    generateVascularField();
  }

  function destroy() {
    if (animId) cancelAnimationFrame(animId);
    window.removeEventListener('resize', resize);
  }

  return { init, destroy };
})();

// Initialize Hero canvas
document.addEventListener('DOMContentLoaded', Hero.init);
// Expose globally
window.Hero = Hero;
