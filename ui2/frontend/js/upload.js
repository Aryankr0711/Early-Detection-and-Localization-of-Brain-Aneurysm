/* ═══════════════════════════════════════════════════════════════
   UPLOAD & PROCESSING STATE MACHINE
   Handles DICOM drag-and-drop, folder browsing, demo cases, and
   simulates pipeline execution with granular log telemetry.
   ═══════════════════════════════════════════════════════════════ */

const Upload = (() => {
  let uploadZone, btnBrowse, btnAnalyze, fileInput, processingContainer, progressBar, processingLog;
  let uploadTitle, uploadSubtitle;
  let isProcessing = false;
  let uploadedFilesData = null;
  let currentCaseData = null;

  // Mock cases database
  const MOCK_CASES = {
    demo: {
      patientId: "RSNA-2025-9928A",
      patientAge: "58 / F",
      patientModality: "CTA",
      inferenceTime: "37.8 seconds",
      inferenceStatus: "CRITICAL FINDING",
      apValue: 0.947,
      apRisk: "CRITICAL",
      telemetry: {
        slices: "186",
        spacing: "0.45mm × 0.45mm × 0.50mm",
        dimensions: "512 × 512 × 186 px",
        orientation: "RAS (Right-Anterior-Superior)",
        modality: "CTA (CT Angiography)",
        totalTime: "37.8s",
        stage1Time: "8.4s",
        stage2Time: "16.2s",
        stage3Time: "13.2s",
        queueTime: "0.2s",
        gpu: "NVIDIA RTX A6000",
        vram: "8.1 GB / 48.0 GB",
        utilization: "94.2%",
        temperature: "68°C",
        version: "v2.4.1-prod",
        classifier: "ResNet3D-Transformer Ensemble",
        segFolds: "3-Fold (0, 1, 2)",
        tta: "Enabled (Flip, Rotate)",
        checkpoint: "epoch-142-auc-0.949"
      },
      vessels: {
        "AComA": { prob: 0.947, ci: "0.91–0.98", risk: "CRITICAL", centerSlice: 93, radius: 14, x: 256, y: 220 },
        "L-MCA": { prob: 0.621, ci: "0.55–0.69", risk: "ELEVATED", centerSlice: 105, radius: 10, x: 195, y: 245 },
        "R-IC Supra": { prob: 0.082, ci: "0.04–0.12", risk: "CLEAR", centerSlice: 85, radius: 0, x: 0, y: 0 },
        "L-IC Supra": { prob: 0.041, ci: "0.01–0.08", risk: "CLEAR", centerSlice: 85, radius: 0, x: 0, y: 0 },
        "R-IC Infra": { prob: 0.012, ci: "0.00–0.03", risk: "CLEAR", centerSlice: 85, radius: 0, x: 0, y: 0 },
        "L-IC Infra": { prob: 0.008, ci: "0.00–0.02", risk: "CLEAR", centerSlice: 85, radius: 0, x: 0, y: 0 },
        "R-MCA": { prob: 0.033, ci: "0.01–0.06", risk: "CLEAR", centerSlice: 85, radius: 0, x: 0, y: 0 },
        "R-ACA": { prob: 0.025, ci: "0.01–0.05", risk: "CLEAR", centerSlice: 85, radius: 0, x: 0, y: 0 },
        "L-ACA": { prob: 0.019, ci: "0.00–0.04", risk: "CLEAR", centerSlice: 85, radius: 0, x: 0, y: 0 },
        "R-PComA": { prob: 0.048, ci: "0.02–0.08", risk: "CLEAR", centerSlice: 85, radius: 0, x: 0, y: 0 },
        "L-PComA": { prob: 0.031, ci: "0.01–0.06", risk: "CLEAR", centerSlice: 85, radius: 0, x: 0, y: 0 },
        "Basilar": { prob: 0.052, ci: "0.02–0.09", risk: "CLEAR", centerSlice: 85, radius: 0, x: 0, y: 0 },
        "Other Post": { prob: 0.029, ci: "0.01–0.05", risk: "CLEAR", centerSlice: 85, radius: 0, x: 0, y: 0 }
      }
    },
    normal: {
      patientId: "RSNA-2025-0194B",
      patientAge: "64 / M",
      patientModality: "MRA",
      inferenceTime: "36.1 seconds",
      inferenceStatus: "CLEAR — NO ANEURYSM DETECTED",
      apValue: 0.024,
      apRisk: "CLEAR",
      telemetry: {
        slices: "160",
        spacing: "0.50mm × 0.50mm × 0.60mm",
        dimensions: "512 × 512 × 160 px",
        orientation: "RAS (Right-Anterior-Superior)",
        modality: "MRA (Magnetic Resonance Angio)",
        totalTime: "36.1s",
        stage1Time: "8.1s",
        stage2Time: "15.4s",
        stage3Time: "12.3s",
        queueTime: "0.3s",
        gpu: "NVIDIA RTX A6000",
        vram: "7.9 GB / 48.0 GB",
        utilization: "91.8%",
        temperature: "65°C",
        version: "v2.4.1-prod",
        classifier: "ResNet3D-Transformer Ensemble",
        segFolds: "3-Fold (0, 1, 2)",
        tta: "Enabled (Flip, Rotate)",
        checkpoint: "epoch-142-auc-0.949"
      },
      vessels: {
        "AComA": { prob: 0.024, ci: "0.01–0.05", risk: "CLEAR", centerSlice: 80, radius: 0, x: 0, y: 0 },
        "L-MCA": { prob: 0.019, ci: "0.00–0.04", risk: "CLEAR", centerSlice: 80, radius: 0, x: 0, y: 0 },
        "R-MCA": { prob: 0.021, ci: "0.01–0.05", risk: "CLEAR", centerSlice: 80, radius: 0, x: 0, y: 0 },
        "R-IC Supra": { prob: 0.008, ci: "0.00–0.02", risk: "CLEAR", centerSlice: 80, radius: 0, x: 0, y: 0 },
        "L-IC Supra": { prob: 0.011, ci: "0.00–0.03", risk: "CLEAR", centerSlice: 80, radius: 0, x: 0, y: 0 },
        "R-IC Infra": { prob: 0.004, ci: "0.00–0.01", risk: "CLEAR", centerSlice: 80, radius: 0, x: 0, y: 0 },
        "L-IC Infra": { prob: 0.005, ci: "0.00–0.01", risk: "CLEAR", centerSlice: 80, radius: 0, x: 0, y: 0 },
        "R-ACA": { prob: 0.014, ci: "0.00–0.03", risk: "CLEAR", centerSlice: 80, radius: 0, x: 0, y: 0 },
        "L-ACA": { prob: 0.012, ci: "0.00–0.03", risk: "CLEAR", centerSlice: 80, radius: 0, x: 0, y: 0 },
        "R-PComA": { prob: 0.007, ci: "0.00–0.02", risk: "CLEAR", centerSlice: 80, radius: 0, x: 0, y: 0 },
        "L-PComA": { prob: 0.009, ci: "0.00–0.02", risk: "CLEAR", centerSlice: 80, radius: 0, x: 0, y: 0 },
        "Basilar": { prob: 0.015, ci: "0.00–0.03", risk: "CLEAR", centerSlice: 80, radius: 0, x: 0, y: 0 },
        "Other Post": { prob: 0.010, ci: "0.00–0.02", risk: "CLEAR", centerSlice: 80, radius: 0, x: 0, y: 0 }
      }
    }
  };

  function init() {
    uploadZone = document.getElementById('uploadZone');
    btnBrowse = document.getElementById('btnBrowse');
    btnAnalyze = document.getElementById('btnAnalyze');
    uploadTitle = document.getElementById('uploadTitle');
    uploadSubtitle = document.getElementById('uploadSubtitle');
    processingContainer = document.getElementById('processingContainer');
    progressBar = document.getElementById('progressBar');
    processingLog = document.getElementById('processingLog');

    // Create file input dynamically
    fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.multiple = true;
    fileInput.webkitdirectory = true;
    fileInput.directory = true;
    fileInput.style.display = 'none';
    document.body.appendChild(fileInput);

    // Bind event listeners
    btnBrowse.addEventListener('click', () => fileInput.click());
    if (btnAnalyze) btnAnalyze.addEventListener('click', startAnalysis);
    fileInput.addEventListener('change', handleFileSelect);

    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);
    uploadZone.addEventListener('click', () => fileInput.click());
  }

  function handleDragOver(e) {
    e.preventDefault();
    if (isProcessing) return;
    uploadZone.classList.add('dragover');
  }

  function handleDragLeave() {
    uploadZone.classList.remove('dragover');
  }

  function handleDrop(e) {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (isProcessing) return;

    const items = e.dataTransfer.items;
    if (items && items.length > 0) {
      const files = [];
      let pending = 0;
      
      function traverseFileTree(item, path) {
        path = path || "";
        if (item.isFile) {
          item.file(function(file) {
            files.push(file);
            pending--;
            if (pending === 0) processFiles(files);
          });
        } else if (item.isDirectory) {
          let dirReader = item.createReader();
          dirReader.readEntries(function(entries) {
            pending--;
            for (let i = 0; i < entries.length; i++) {
              pending++;
              traverseFileTree(entries[i], path + item.name + "/");
            }
            if (pending === 0) processFiles(files);
          });
        }
      }
      
      for (let i = 0; i < items.length; i++) {
        let item = items[i].webkitGetAsEntry();
        if (item) {
          pending++;
          traverseFileTree(item);
        }
      }
    } else {
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        processFiles(files);
      }
    }
  }

  function handleFileSelect(e) {
    if (isProcessing) return;
    const files = e.target.files;
    if (files.length > 0) {
      processFiles(files);
    }
  }

  function processFiles(files) {
    let fileCount = files.length;
    let patientName = "ANON-" + Math.floor(10000 + Math.random() * 90000);

    // Create a base template to hold the real results
    const customCase = JSON.parse(JSON.stringify(MOCK_CASES.demo));
    customCase.patientId = patientName;
    customCase.telemetry.slices = String(fileCount);

    uploadedFilesData = files;
    currentCaseData = customCase;

    // Update UI for two-step process
    if (uploadTitle) uploadTitle.textContent = "Upload Complete";
    if (uploadSubtitle) uploadSubtitle.textContent = `${fileCount} DICOM files ready for analysis`;
    if (uploadZone) {
      uploadZone.style.border = "2px solid var(--accent)";
      uploadZone.style.backgroundColor = "rgba(0, 180, 216, 0.05)";
    }
    
    btnBrowse.style.display = "none";
    if (btnAnalyze) btnAnalyze.style.display = "inline-block";
  }

  function startAnalysis() {
    if (!uploadedFilesData || !currentCaseData) return;
    if (btnAnalyze) {
      btnAnalyze.disabled = true;
      btnAnalyze.textContent = "Analyzing...";
    }
    runInferencePipeline(currentCaseData, uploadedFilesData);
  }


  function logMessage(text, type = "info") {
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false, fractionalSecondDigits: 3 });
    const entry = document.createElement('div');
    entry.className = 'log-entry active';

    let prefix = '<span class="status-run">[RUN]</span>';
    if (type === "ok") prefix = '<span class="status-ok">[ OK ]</span>';
    if (type === "fail") prefix = '<span class="status-fail">[FAIL]</span>';
    if (type === "info") prefix = '<span class="status-info">[INFO]</span>';

    entry.innerHTML = `<span class="timestamp">${timestamp}</span> ${prefix} <span>${text}</span>`;
    
    // Dim previous active log entries
    const activeEntries = processingLog.querySelectorAll('.log-entry.active');
    activeEntries.forEach(el => el.classList.remove('active'));

    processingLog.appendChild(entry);
    processingLog.scrollTop = processingLog.scrollHeight;
  }

  function runInferencePipeline(caseData, files = null) {
    isProcessing = true;
    btnBrowse.disabled = true;
    uploadZone.style.pointerEvents = 'none';
    uploadZone.style.opacity = '0.5';

    // Hide any previous results
    const resultsSec = document.getElementById('results');
    const viewerSec = document.getElementById('viewer');
    resultsSec.classList.remove('visible');
    if (viewerSec) viewerSec.classList.remove('visible');

    processingContainer.classList.add('active');
    processingLog.innerHTML = "";
    progressBar.style.width = "0%";

    // Start background fetch if files are provided
    let backendPromise = null;
    if (files) {
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
      }
      backendPromise = fetch('/api/analyze', {
        method: 'POST',
        body: formData
      }).then(res => res.json()).catch(err => {
        return { error: err.message };
      });
    }

    const timeline = [
      { progress: 0, text: "DICOM ingestion protocol initialized. Reading files...", type: "info", stage: 0 },
      { progress: 5, text: "Orientation detected: axial scan matrix. Standardizing to RAS orientation...", type: "info", stage: 0 },
      { progress: 10, text: "RAS alignment complete. Checking header data...", type: "info", stage: 0 },
      { progress: 15, text: "Stage 01: Spatially-Sparse search initialized (nnU-Net 3D search fold 0)...", type: "run", stage: 1 },
      { progress: 22, text: "Stage 01: Multi-scale vessel enhancement filtering active...", type: "info", stage: 1 },
      { progress: 28, text: "Stage 01: DBSCAN centroid extraction calculating coordinates...", type: "info", stage: 1 },
      { progress: 35, text: "Stage 01: Sparse search complete. Centroid locked: X:124.2, Y:158.4, Z:94.0", type: "ok", stage: 1 },
      { progress: 40, text: "Adaptive 3D crop bounding box extracted (140×140×140 mm ROI).", type: "info", stage: 1 },
      { progress: 45, text: "Stage 02: Spatially-Dense segmentation models initializing...", type: "run", stage: 2 },
      { progress: 52, text: "Stage 02: Model A (SkeletonRecall loss) inference active...", type: "info", stage: 2 },
      { progress: 60, text: "Stage 02: Model B (FocalTversky loss) inference active...", type: "info", stage: 2 },
      { progress: 68, text: "Stage 02: Resolving ensemble consensus voting (13-channel mapping)...", type: "info", stage: 2 },
      { progress: 75, text: "Stage 02: Segmentation complete. Refined 3D vessel segmentations compiled.", type: "ok", stage: 2 },
      { progress: 80, text: "Stage 03: RegionMaskedPooling3D localization classifier initializing...", type: "run", stage: 3 },
      { progress: 85, text: "Stage 03: Extracting feature representations from localized region...", type: "info", stage: 3 },
      { progress: 90, text: "Stage 03: Running 5-fold ensemble with Test-Time Augmentation (TTA)...", type: "info", stage: 3 },
      { progress: 95, text: "Stage 03: Probability vectors evaluated for 13 location heads + AP presence head.", type: "ok", stage: 3 },
      { progress: 99, text: "Pipeline complete. Post-processing prediction thresholds standardizing...", type: "info", stage: 0 }
    ];

    let step = 0;
    function executeStep() {
      if (step < timeline.length) {
        const item = timeline[step];
        progressBar.style.width = item.progress + "%";
        logMessage(item.text, item.type);

        if (item.stage > 0) {
          Pipeline.setStage(item.stage);
        }

        step++;
        setTimeout(executeStep, 200 + Math.random() * 150);
      } else {
        if (backendPromise) {
            logMessage("Waiting for real backend inference to complete... (This may take ~30s on GPU)", "run");
            
            backendPromise.then(result => {
                if (result.error) {
                    logMessage(`Backend Error: ${result.error}`, "fail");
                    progressBar.style.width = "100%";
                    progressBar.style.backgroundColor = "var(--status-critical)";
                    resetUI();
                } else {
                    logMessage("Backend inference complete. Results received.", "ok");
                    progressBar.style.width = "100%";
                    
                    if (result.patientId) caseData.patientId = result.patientId;
                    if (result.patientAge) caseData.patientAge = result.patientAge;
                    if (result.patientModality) {
                        caseData.patientModality = result.patientModality;
                        caseData.telemetry.modality = result.patientModality;
                    }
                    caseData.apValue = result.apValue;
                    caseData.apRisk = result.apRisk;
                    caseData.inferenceTime = result.inferenceTime + " seconds";
                    caseData.inferenceStatus = result.apRisk === "CLEAR" ? "CLEAR — NO ANEURYSM DETECTED" : "CRITICAL FINDING";
                    caseData.telemetry.totalTime = result.telemetry.totalTime;
                    
                    for (const v in result.vessels) {
                        if (caseData.vessels[v]) {
                            const probability = Number(result.vessels[v]);
                            caseData.vessels[v].prob = probability;
                            caseData.vessels[v].risk = probability > 0.7 ? "CRITICAL" : probability > 0.3 ? "ELEVATED" : "CLEAR";
                        }
                    }
                    finalize(caseData);
                }
            });
        } else {
            progressBar.style.width = "100%";
            finalize(caseData);
        }
      }
    }
    
    function resetUI() {
        isProcessing = false;
        btnBrowse.disabled = false;
        btnBrowse.style.display = "inline-block";
        if (btnAnalyze) {
          btnAnalyze.style.display = "none";
          btnAnalyze.disabled = false;
          btnAnalyze.textContent = "Start Analysis";
        }
        if (uploadTitle) uploadTitle.textContent = "Drop DICOM Series";
        if (uploadSubtitle) uploadSubtitle.textContent = "or click to browse · supports .dcm directories";
        if (uploadZone) {
          uploadZone.style.border = "";
          uploadZone.style.backgroundColor = "";
          uploadZone.style.pointerEvents = 'auto';
          uploadZone.style.opacity = '1';
        }
        uploadedFilesData = null;
        currentCaseData = null;
    }

    function finalize(finalData) {
        setTimeout(() => {
          logMessage("Diagnostic report generated. Rendering interactive workstation views...", "ok");
          
          setTimeout(() => {
            Results.loadResults(finalData);
            resetUI();
            resultsSec.scrollIntoView({ behavior: 'smooth' });
          }, 300);
        }, 500);
    }

    executeStep();
  }

  // Initialize
  document.addEventListener('DOMContentLoaded', init);

  return {
    runInferencePipeline
  };
})();

// Expose globally
window.Upload = Upload;
