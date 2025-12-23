#!/usr/bin/env python3
"""
Sync reports from gso-internal to gso-experiments.
Reports are copied as-is; metadata tracked in manifest.json.
"""

import json
import shutil
from pathlib import Path

DOCENT_BASE = "https://docent.transluce.org/dashboard"
GCS_BASE = "gs://gso-experiments"


def main():
    repo_dir = Path(__file__).parent.parent
    models_json = repo_dir / "models.json"
    reports_src = Path.home() / "gso-internal" / "reports"
    reports_dst = repo_dir / "results" / "reports"
    reports_dst.mkdir(parents=True, exist_ok=True)

    with open(models_json) as f:
        models = json.load(f)

    manifest = {"models": {}}

    for model_name, cfg in models.items():
        report_file = cfg["report_file"]
        docent_id = cfg.get("docent_id", "")

        src_path = reports_src / report_file
        dst_path = reports_dst / f"{model_name}.json"

        if not src_path.exists():
            print(f"SKIP {model_name}: {src_path} not found")
            continue

        # Copy report as-is
        shutil.copy(src_path, dst_path)
        print(f"OK {model_name}: {dst_path.name}")

        # Read summary for manifest
        with open(src_path) as f:
            report = json.load(f)

        summary = report.get("summary", {})
        manifest["models"][model_name] = {
            "docent_url": f"{DOCENT_BASE}/{docent_id}" if docent_id else None,
            "gcs_path": f"{GCS_BASE}/{model_name}",
            "report": f"reports/{model_name}.json",
            "summary": {
                "total_instances": summary.get("total_instances"),
                "opt_commit": summary.get("opt_commit"),
                "opt_base": summary.get("opt_base"),
                "passed": summary.get("passed_instances"),
                "score": summary.get("score"),
            },
        }

    # Write manifest
    manifest_path = repo_dir / "results" / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
