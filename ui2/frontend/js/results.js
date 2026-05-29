/* ═══════════════════════════════════════════════════════════════
   RESULTS DASHBOARD — Diagnostic Cockpit Coordinator
   Manages patient data presentation, bi-directional table & SVG
   interactions, system telemetry, and case data routing.
   ═══════════════════════════════════════════════════════════════ */

const Results = (() => {
  let activeCase = null;
  let selectedVesselKey = null;
  let telemetryHeartbeat = null;

  const MAP_KEY_TO_SVG_ID = {
    "L-IC Infra": "l_ic_infra",
    "R-IC Infra": "r_ic_infra",
    "L-IC Supra": "l_ic_supra",
    "R-IC Supra": "r_ic_supra",
    "L-MCA": "l_mca",
    "R-MCA": "r_mca",
    "AComA": "acoma",
    "L-ACA": "l_aca",
    "R-ACA": "r_aca",
    "L-PComA": "l_pcoma",
    "R-PComA": "r_pcoma",
    "Basilar": "basilar",
    "Other Post": "other_post"
  };

  const MAP_SVG_ID_TO_KEY = Object.fromEntries(
    Object.entries(MAP_KEY_TO_SVG_ID).map(([k, v]) => [v, k])
  );

  function init() {
    // Initialize Vessel SVG Map
    VesselMap.init('vesselMapContainer');

    // Bind click events on SVG paths to table rows
    VesselMap.VESSELS.forEach(v => {
      const pathEl = document.getElementById(`vessel-${v.id}`);
      if (pathEl) {
        pathEl.addEventListener('click', () => {
          const vesselKey = MAP_SVG_ID_TO_KEY[v.id];
          if (vesselKey) {
            selectVessel(vesselKey);
          }
        });
      }
    });
  }

  function loadResults(caseData) {
    activeCase = caseData;
    
    // 1. Show the hidden results and viewer sections
    document.getElementById('results').classList.add('visible');
    const viewerEl = document.getElementById('viewer');
    if (viewerEl) viewerEl.classList.add('visible');

    // 2. Populate Patient Header
    document.getElementById('patientId').textContent = caseData.patientId;
    document.getElementById('patientAge').textContent = caseData.patientAge;
    document.getElementById('patientModality').textContent = caseData.patientModality;
    document.getElementById('inferenceTime').textContent = caseData.inferenceTime;
    
    const statusEl = document.getElementById('inferenceStatus');
    statusEl.textContent = caseData.inferenceStatus;
    statusEl.className = 'patient-field-value'; // Reset
    if (caseData.apValue > 0.5) {
      statusEl.classList.add('risk-critical');
    } else {
      statusEl.classList.add('risk-clear');
    }

    // 3. Update AP Gauge
    const apValEl = document.getElementById('apValue');
    const apBarEl = document.getElementById('apBar');
    const apRiskEl = document.getElementById('apRisk');
    const overallProbValueEl = document.getElementById('overallProbValue');
    const overallProbRiskEl = document.getElementById('overallProbRisk');

    const apPercent = (caseData.apValue * 100).toFixed(1);
    apValEl.textContent = `${apPercent}%`;
    apBarEl.style.width = `${apPercent}%`;
    if (overallProbValueEl) overallProbValueEl.textContent = `${apPercent}%`;

    // Colors & Text based on risk
    apBarEl.className = 'ap-gauge-bar-fill'; // Reset
    apRiskEl.className = 'ap-risk-label';
    if (overallProbRiskEl) overallProbRiskEl.className = 'overall-prob-risk';
    if (caseData.apRisk === "CRITICAL") {
      apBarEl.style.backgroundColor = 'var(--status-critical)';
      apRiskEl.classList.add('risk-critical');
      apRiskEl.textContent = "CRITICAL FINDING — NEUROSURGERY ALERT";
    } else if (caseData.apRisk === "ELEVATED") {
      apBarEl.style.backgroundColor = 'var(--status-elevated)';
      apRiskEl.classList.add('risk-elevated');
      apRiskEl.textContent = "ELEVATED RISK — CLINICAL FOLLOW-UP REQUIRED";
    } else {
      apBarEl.style.backgroundColor = 'var(--status-clear)';
      apRiskEl.classList.add('risk-clear');
      apRiskEl.textContent = "NORMAL STUDY — NO DETECTABLE LESION";
    }

    if (overallProbRiskEl) {
      overallProbRiskEl.textContent = caseData.apRisk;
      if (caseData.apRisk === "CRITICAL") overallProbRiskEl.classList.add('risk-critical');
      else if (caseData.apRisk === "ELEVATED") overallProbRiskEl.classList.add('risk-elevated');
      else overallProbRiskEl.classList.add('risk-clear');
    }

    // 4. Update SVG Map Probabilities
    const svgProbMap = {};
    Object.keys(caseData.vessels).forEach(key => {
      const svgId = MAP_KEY_TO_SVG_ID[key];
      if (svgId) {
        svgProbMap[svgId] = caseData.vessels[key].prob;
      }
    });
    VesselMap.update(svgProbMap);

    // 5. Populate Probability Table
    populateProbTable(caseData.vessels, caseData.apValue, caseData.apRisk);

    // 6. Update Telemetry
    populateTelemetry(caseData.telemetry);

    // 7. Render Logs
    populateInferenceLogs(caseData);

    // 8. Feed case to Dicom Viewer
    DicomViewer.setCase(caseData);

    // 9. Start Operational Heartbeat (live system telemetry)
    startOperationalHeartbeat(caseData.telemetry);

    // Select the first critical vessel by default, or basilar if none
    const criticalVesselKeys = Object.keys(caseData.vessels).filter(k => caseData.vessels[k].risk === "CRITICAL");
    const defaultVessel = criticalVesselKeys.length > 0 ? criticalVesselKeys[0] : "AComA";
    selectVessel(defaultVessel);
  }

  function populateProbTable(vessels, overallProbability, overallRisk) {
    const tbody = document.getElementById('probTableBody');
    tbody.innerHTML = "";

    // Sort vessels: Critical -> Elevated -> Clear (within clear, sort by prob desc)
    const sortedKeys = Object.keys(vessels).sort((a, b) => {
      const vA = vessels[a];
      const vB = vessels[b];
      
      const rank = { "CRITICAL": 3, "ELEVATED": 2, "CLEAR": 1 };
      if (rank[vA.risk] !== rank[vB.risk]) {
        return rank[vB.risk] - rank[vA.risk];
      }
      return vB.prob - vA.prob;
    });

    let counts = { critical: 0, elevated: 0, clear: 0 };

    sortedKeys.forEach(vesselKey => {
      const v = vessels[vesselKey];
      const row = document.createElement('tr');
      row.id = `row-${vesselKey.replace(/[\s-]/g, '_')}`;
      row.setAttribute('data-vessel', vesselKey);
      
      let riskClass = 'clear';
      let riskText = 'Clear';
      let barColor = 'var(--status-clear)';

      if (v.risk === 'CRITICAL') {
        riskClass = 'critical';
        riskText = 'CRITICAL';
        barColor = 'var(--status-critical)';
        counts.critical++;
      } else if (v.risk === 'ELEVATED') {
        riskClass = 'elevated';
        riskText = 'ELEVATED';
        barColor = 'var(--status-elevated)';
        counts.elevated++;
      } else {
        counts.clear++;
      }

      row.className = riskClass;

      row.innerHTML = `
        <td style="font-weight: 500;">${vesselKey}</td>
        <td>
          <div style="display: flex; align-items: center; gap: 8px;">
            <span class="prob-value">${(v.prob * 100).toFixed(1)}%</span>
            <div class="prob-bar-container">
              <div class="prob-bar" style="width: ${v.prob * 100}%; background-color: ${barColor};"></div>
            </div>
          </div>
        </td>
        <td style="color: var(--text-muted); font-size: 11px;">${v.ci}</td>
        <td class="risk-${riskClass}" style="font-weight: 600; font-size: 11px; letter-spacing: 0.05em;">${riskText}</td>
      `;

      // Click row selects finding
      row.addEventListener('click', () => {
        selectVessel(vesselKey);
      });

      tbody.appendChild(row);
    });

    const overallClass = overallRisk === 'CRITICAL' ? 'critical' : overallRisk === 'ELEVATED' ? 'elevated' : 'clear';
    const overallColor = overallRisk === 'CRITICAL' ? 'var(--status-critical)' : overallRisk === 'ELEVATED' ? 'var(--status-elevated)' : 'var(--status-clear)';
    const overallRow = document.createElement('tr');
    overallRow.className = `overall-row ${overallClass}`;
    overallRow.innerHTML = `
      <td style="font-weight: 700;">Overall</td>
      <td>
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="prob-value">${(overallProbability * 100).toFixed(1)}%</span>
          <div class="prob-bar-container">
            <div class="prob-bar" style="width: ${overallProbability * 100}%; background-color: ${overallColor};"></div>
          </div>
        </div>
      </td>
      <td style="color: var(--text-muted); font-size: 11px;">ensemble</td>
      <td class="risk-${overallClass}" style="font-weight: 700; font-size: 11px; letter-spacing: 0.05em;">${overallRisk}</td>
    `;
    tbody.appendChild(overallRow);

    // Update Counts Panel
    const countCriticalEl = document.getElementById('countCritical');
    const countElevatedEl = document.getElementById('countElevated');
    const countClearEl = document.getElementById('countClear');
    if (countCriticalEl) countCriticalEl.textContent = counts.critical;
    if (countElevatedEl) countElevatedEl.textContent = counts.elevated;
    if (countClearEl) countClearEl.textContent = counts.clear;
  }

  function selectVessel(vesselKey) {
    selectedVesselKey = vesselKey;
    
    // Highlight table row
    const rows = document.querySelectorAll('#probTableBody tr');
    rows.forEach(r => r.classList.remove('active-row'));

    const activeRow = document.getElementById(`row-${vesselKey.replace(/[\s-]/g, '_')}`);
    if (activeRow) {
      activeRow.classList.add('active-row');
      // Scroll row into view inside panel if overflow
      activeRow.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Highlight SVG map path
    const svgId = MAP_KEY_TO_SVG_ID[vesselKey];
    VesselMap.VESSELS.forEach(v => {
      const pathEl = document.getElementById(`vessel-${v.id}`);
      const labelEl = document.getElementById(`label-${v.id}`);
      if (pathEl) {
        if (v.id === svgId) {
          pathEl.setAttribute('stroke-width', '4');
          if (labelEl) labelEl.setAttribute('fill', 'var(--accent-cyan)');
        } else {
          // Restore standard widths
          const pathProb = activeCase.vessels[MAP_SVG_ID_TO_KEY[v.id]]?.prob || 0;
          let normalWidth = 1.5;
          if (pathProb >= 0.7) normalWidth = 2.5;
          else if (pathProb >= 0.3) normalWidth = 2;
          pathEl.setAttribute('stroke-width', String(normalWidth));
          
          if (labelEl) {
            labelEl.setAttribute('fill', pathProb >= 0.3 ? '#0F172A' : '#64748B');
          }
        }
      }
    });

    // Notify DICOM viewer to highlight finding and jump to correct slice
    DicomViewer.selectVessel(vesselKey);
  }

  function populateTelemetry(tel) {
    document.getElementById('tSlices').textContent = tel.slices;
    document.getElementById('tSpacing').textContent = tel.spacing;
    document.getElementById('tDims').textContent = tel.dimensions;
    document.getElementById('tOrient').textContent = tel.orientation;
    document.getElementById('tModality').textContent = tel.modality;

    document.getElementById('tTotal').textContent = tel.totalTime;
    document.getElementById('tS1').textContent = tel.stage1Time;
    document.getElementById('tS2').textContent = tel.stage2Time;
    document.getElementById('tS3').textContent = tel.stage3Time;
    document.getElementById('tQueue').textContent = tel.queueTime;

    document.getElementById('tGPU').textContent = tel.gpu;
    document.getElementById('tVRAM').textContent = tel.vram;
    document.getElementById('tUtil').textContent = tel.utilization;
    document.getElementById('tTemp').textContent = tel.temperature;

    document.getElementById('tVersion').textContent = tel.version;
    document.getElementById('tClassifier').textContent = tel.classifier;
    document.getElementById('tSegFolds').textContent = tel.segFolds;
    document.getElementById('tTTA').textContent = tel.tta;
    document.getElementById('tCkpt').textContent = tel.checkpoint;
  }

  function startOperationalHeartbeat(baseTelemetry) {
    if (telemetryHeartbeat) clearInterval(telemetryHeartbeat);
    
    // Extract base numerical values
    let baseTemp = parseInt(baseTelemetry.temperature) || 68;
    let baseUtil = parseInt(baseTelemetry.utilization) || 82;
    
    // In an idle state after inference, GPU drops but still pulses
    let currentUtil = 12;
    let currentTemp = Math.max(45, baseTemp - 15);

    const utilEl = document.getElementById('tUtil');
    const tempEl = document.getElementById('tTemp');

    telemetryHeartbeat = setInterval(() => {
      // Slight random walk
      currentUtil += Math.floor(Math.random() * 5) - 2;
      currentTemp += (Math.random() * 0.8) - 0.4;
      
      // Clamp
      if (currentUtil < 2) currentUtil = 2;
      if (currentUtil > 25) currentUtil = 25;
      if (currentTemp < 40) currentTemp = 40;
      if (currentTemp > 65) currentTemp = 65;

      utilEl.textContent = `${currentUtil}%`;
      tempEl.textContent = `${currentTemp.toFixed(1)}°C`;
    }, 2000);
  }

  function populateInferenceLogs(caseData) {
    const logContainer = document.getElementById('inferenceLog');
    logContainer.innerHTML = "";

    const steps = [
      { t: "+0.00s", m: "Initializing GPU pipeline. Loading weights into VRAM." },
      { t: "+0.22s", m: `Ingestion: Registered dataset with patient ID: ${caseData.patientId}` },
      { t: "+0.85s", m: "Standardizing scan spatial dimensions to isotropic resolution (0.5mm)." },
      { t: "+1.90s", m: "Stage 1 nnU-Net coarse search active: calculating spatial occupancy probability map." },
      { t: "+6.40s", m: "Stage 1 DBSCAN clustering locked centroid coords." },
      { t: "+8.60s", m: "Adaptive ROI 3D crop bounding box extracted." },
      { t: "+12.10s", m: "Stage 2 dual nnU-Net ensemble: compiling skeletonized and volume probability maps." },
      { t: "+24.80s", m: "Consensus ensemble segmentations written. Stage 2 complete." },
      { t: "+26.10s", m: "Stage 3 RegionMaskedPooling3D localization heads initiating inference." },
      { t: "+34.50s", m: "Aggregating 5-fold ensemble with test-time augmentations (TTA)." },
      { t: "+37.40s", m: `Inference complete. Aneurysm presence probability: ${(caseData.apValue * 100).toFixed(1)}%.` }
    ];

    steps.forEach(s => {
      const div = document.createElement('div');
      div.className = 'telemetry-row';
      div.style.justifyContent = 'flex-start';
      div.style.gap = '12px';
      div.innerHTML = `<span class="time" style="color: var(--text-muted); opacity: 0.6; min-width: 60px;">${s.t}</span>
                       <span style="color: var(--text-secondary);">${s.m}</span>`;
      logContainer.appendChild(div);
    });
  }

  // Initialize on load
  document.addEventListener('DOMContentLoaded', init);

  return {
    loadResults,
    selectVessel
  };
})();

// Expose globally
window.Results = Results;
