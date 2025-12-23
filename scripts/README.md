# GSO Experiments

Trajectories, logs, and results from GSO benchmark evaluations.

## Setup

```bash
uv sync
cp .env.example .env  # Add your DOCENT_API_KEY
```

## Push Results

```bash
# Push to GCS only
./scripts/push.sh gcs

# Push to Docent only
./scripts/push.sh docent

# Push to both GCS and Docent
./scripts/push.sh all
```

## Add a New Model

Edit `models.json` to add a new entry:

```json
"model-name": {
  "trajs_dir": "model-name_maxiter_100_N_v0.51.1-no-hint-run_1",
  "logs_dir": "model-name_maxiter_100_N_v0.51.1-no-hint-run_1",
  "report_file": "model-name.opt@1.pass.report.json"
}
```

Then run `./scripts/push.sh all`. The `docent_id` is auto-populated after upload.

## Pull from GCS

```bash
gsutil ls gs://gso-experiments/
gsutil -m cp -r gs://gso-experiments/<model> ./
```

## Results

- `results/manifest.json` - Model summary with Docent URLs
- `results/reports/<model>.json` - Full evaluation reports
