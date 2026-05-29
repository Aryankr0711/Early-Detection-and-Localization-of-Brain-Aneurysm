# RSNA UI2 Web App

This folder contains a self-contained web app for the RSNA 2025 aneurysm project.
It does not modify the existing model code. The backend imports the current repo
inference code when the full Python/GPU environment is available, and falls back
to deterministic demo output when dependencies are missing.

## Run

```powershell
cd /d E:\rsna2025_main
py ui2\server.py
```

Open:

```text
http://127.0.0.1:7860
```

Optional port override:

```powershell
$env:UI2_PORT="7870"; py ui2\server.py
```

## API

- `GET /api/health`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/result`
- `POST /api/jobs/path`
- `POST /api/jobs/upload`

Example path request:

```json
{
  "path": "E:\\rsna2025_main\\sample_data",
  "mode": "auto",
  "save_roi": false,
  "save_seg": false
}
```

Use `"mode": "demo"` to force fast UI testing without loading the model stack.
