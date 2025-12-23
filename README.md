# GSO Experiments

Results from model evaluations on [GSO (Global Software Optimization)](https://gso-bench.github.io/).

## Structure

```
gso-experiments/
├── models.json              # Model configurations
├── results/
│   ├── manifest.json        # Evaluation summaries and Docent URLs to agent trajectories
│   └── reports/             # Full evaluation reports from GSO
│       └── <model>.json
└── scripts/                 # Internal tooling
```

## Viewing Agent Trajectories and Logs

Agent trajectories are viewable via [Docent](https://transluce.org/docent), a tool by Transluce for exploring and analyzing agent behavior.

- **Trajectories**: See Docent URLs in `results/manifest.json` to browse agent runs
- **Reports**: See `results/reports/<model>.json` for evaluation reports
- **Raw data**: `gsutil -m cp -r gs://gso-experiments/<model> ./`

We recommend using Docent for most analysis cases. However, if you need to download raw data, you can use the option below.

### Downloading Raw Data from GCS

Raw **trajectories and evaluation logs** are stored in a public [Google Cloud Storage](https://cloud.google.com/storage) bucket. To download them, you'll need to:

1. [Create a Google Cloud account](https://cloud.google.com/free)
2. [Install the Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
3. Run `gcloud init` to configure your credentials

Then download data with:
```bash
# List available models
gsutil ls gs://gso-experiments/

# Download a specific model's data
gsutil -m cp -r gs://gso-experiments/<model> ./
```

Each model's data is stored in the following structure:
```
gs://gso-experiments/<model>/
├── trajs/                   # Agent trajectories
│   ├── output.jsonl
│   └── metadata.json
├── logs/                    # GSO evaluation logs
└── report.json              # Aggregated results
```

## Submissions

> [!TODO] Add instructions for submitting results for a new model


## License

MIT
