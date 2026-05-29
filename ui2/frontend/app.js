const state = {
  activeJobId: null,
  pollTimer: null,
};

const stages = [
  "input validation",
  "DICOM audit",
  "majority slice filtering",
  "NIfTI conversion",
  "sparse vessel search",
  "dense segmentation",
  "ROI classification",
  "completed",
];

const labels = [
  "Left ICA",
  "Right ICA",
  "Left ACA",
  "Right ACA",
  "Left MCA",
  "Right MCA",
  "Left PCA",
  "Right PCA",
  "Left PCOM",
  "Right PCOM",
  "Left VA",
  "Right VA",
  "Basilar",
  "Aneurysm Present",
];

const els = {
  healthChip: document.querySelector("#healthChip"),
  pathForm: document.querySelector("#pathForm"),
  dicomPath: document.querySelector("#dicomPath"),
  demoMode: document.querySelector("#demoMode"),
  saveRoi: document.querySelector("#saveRoi"),
  saveSeg: document.querySelector("#saveSeg"),
  uploadZone: document.querySelector("#uploadZone"),
  fileInput: document.querySelector("#fileInput"),
  jobStatus: document.querySelector("#jobStatus"),
  jobStage: document.querySelector("#jobStage"),
  progressBar: document.querySelector("#progressBar"),
  progressLabel: document.querySelector("#progressLabel"),
  timeline: document.querySelector("#timeline"),
  probabilityList: document.querySelector("#probabilityList"),
  logOutput: document.querySelector("#logOutput"),
  refreshButton: document.querySelector("#refreshButton"),
  riskCircle: document.querySelector("#riskCircle"),
  riskValue: document.querySelector("#riskValue"),
  riskLabel: document.querySelector("#riskLabel"),
  sourceBadge: document.querySelector("#sourceBadge"),
  activeModeLabel: document.querySelector("#activeModeLabel"),
};

function init() {
  renderTimeline("queued");
  renderEmptyProbabilities();
  checkHealth();
  els.pathForm.addEventListener("submit", onPathSubmit);
  els.demoMode.addEventListener("change", () => {
    els.activeModeLabel.textContent = els.demoMode.checked ? "Demo inference" : "Auto inference";
  });
  els.refreshButton.addEventListener("click", () => {
    if (state.activeJobId) pollJob();
  });
  setupUpload();
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
    });
  });
}

async function checkHealth() {
  try {
    const data = await getJson("/api/health");
    els.healthChip.textContent = data.ok ? "backend online" : "backend issue";
  } catch {
    els.healthChip.textContent = "backend offline";
  }
}

async function onPathSubmit(event) {
  event.preventDefault();
  const path = els.dicomPath.value.trim();
  if (!path) return;
  const payload = {
    path,
    mode: els.demoMode.checked ? "demo" : "auto",
    save_roi: els.saveRoi.checked,
    save_seg: els.saveSeg.checked,
  };
  const data = await postJson("/api/jobs/path", payload);
  attachJob(data.job.id);
}

function setupUpload() {
  els.uploadZone.addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", () => uploadFiles(els.fileInput.files));
  ["dragenter", "dragover"].forEach((name) => {
    els.uploadZone.addEventListener(name, (event) => {
      event.preventDefault();
      els.uploadZone.classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    els.uploadZone.addEventListener(name, (event) => {
      event.preventDefault();
      els.uploadZone.classList.remove("dragging");
    });
  });
  els.uploadZone.addEventListener("drop", (event) => uploadFiles(event.dataTransfer.files));
}

async function uploadFiles(files) {
  if (!files || files.length === 0) return;
  const formData = new FormData();
  [...files].forEach((file) => formData.append("files", file, file.webkitRelativePath || file.name));
  const mode = els.demoMode.checked ? "demo" : "auto";
  const response = await fetch(`/api/jobs/upload?mode=${encodeURIComponent(mode)}`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) throw new Error(await response.text());
  const data = await response.json();
  attachJob(data.job.id);
}

function attachJob(jobId) {
  state.activeJobId = jobId;
  clearInterval(state.pollTimer);
  pollJob();
  state.pollTimer = setInterval(pollJob, 900);
}

async function pollJob() {
  if (!state.activeJobId) return;
  const data = await getJson(`/api/jobs/${state.activeJobId}`);
  renderJob(data.job);
  if (["completed", "failed"].includes(data.job.status)) {
    clearInterval(state.pollTimer);
  }
}

function renderJob(job) {
  els.jobStatus.textContent = `${job.status.toUpperCase()} · ${job.id}`;
  els.jobStage.textContent = job.error || job.stage || "working";
  const progress = Number(job.progress || 0);
  els.progressBar.style.width = `${progress}%`;
  els.progressLabel.textContent = `${progress}%`;
  renderTimeline(job.stage);
  els.logOutput.textContent = (job.logs || []).join("\n") || "No logs yet.";
  els.logOutput.scrollTop = els.logOutput.scrollHeight;
  if (job.result) {
    renderResult(job.result);
  }
}

function renderTimeline(activeStage) {
  const activeIndex = Math.max(0, stages.findIndex((stage) => stage === activeStage));
  els.timeline.innerHTML = stages
    .map((stage, index) => `<li class="${index <= activeIndex ? "done" : ""}">${stage}</li>`)
    .join("");
}

function renderEmptyProbabilities() {
  els.probabilityList.innerHTML = labels
    .map(
      (label) => `
        <div class="prob-row">
          <span>${label}</span>
          <div class="bar"><span style="width:0%"></span></div>
          <strong>--</strong>
        </div>
      `,
    )
    .join("");
}

function renderResult(result) {
  els.sourceBadge.textContent = result.source === "real" ? "trained model" : "demo fallback";
  const present = Number(result.aneurysm_present_probability || 0);
  const percent = Math.round(present * 100);
  els.riskValue.textContent = `${percent}%`;
  els.riskLabel.textContent = `${result.risk_level || "Unknown"} risk`;
  const circumference = 439.82;
  els.riskCircle.style.strokeDashoffset = `${circumference - circumference * present}`;
  els.riskCircle.style.stroke = present >= 0.7 ? "var(--danger)" : present >= 0.35 ? "var(--risk)" : "var(--ok)";

  els.probabilityList.innerHTML = result.probabilities
    .map((item) => {
      const value = Number(item.value || 0);
      const width = Math.round(value * 100);
      const color = value >= 0.7 ? "var(--danger)" : value >= 0.35 ? "var(--risk)" : "var(--accent)";
      return `
        <div class="prob-row">
          <span>${item.label}</span>
          <div class="bar"><span style="width:${width}%; background:${color}"></span></div>
          <strong>${width}%</strong>
        </div>
      `;
    })
    .join("");
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

init();
