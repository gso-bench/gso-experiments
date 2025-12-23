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

We recommend using the Docent for most analysis cases. 
However raw **trajectories and evaluation logs** are stored in Google Cloud Storage in the following structure:
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
