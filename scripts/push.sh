#!/bin/bash
# Push GSO evaluation results to GCS and/or Docent
#
# Usage:
#   ./scripts/push.sh gcs         # Push to GCS only
#   ./scripts/push.sh docent     # Push to Docent only
#   ./scripts/push.sh all        # Push to both (default)
#   ./scripts/push.sh docent new-only   # Docent: only models without docent_id (keeps existing links)
#   ./scripts/push.sh all new-only      # GCS + Docent for new models only

set +e  # Don't exit on errors (some symlinks may fail)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
BUCKET="gs://gso-experiments"
MODELS_JSON="$REPO_DIR/models.json"

# Source directories
TRAJS_BUCKET="$HOME/buckets/gso_bucket/submissions/gso-bench__gso-test/CodeActAgent/pass"
TRAJS_OH="$HOME/OpenHands/evaluation/evaluation_outputs/outputs/gso-bench__gso-test/CodeActAgent/pass"
TRAJS_SCAFFOLDS_OH="$HOME/scaffolds/openhands-runs"
LOGS_BASE="$HOME/gso-internal/logs/run_evaluation/pass"
REPORTS_DIR="$HOME/gso-internal/reports"

# Track Docent collection IDs for summary
declare -A DOCENT_COLLECTIONS

find_trajs_path() {
    local trajs_dir=$1
    if [ -d "$TRAJS_SCAFFOLDS_OH/$trajs_dir" ]; then
        echo "$TRAJS_SCAFFOLDS_OH/$trajs_dir"
    elif [ -d "$TRAJS_BUCKET/$trajs_dir" ]; then
        echo "$TRAJS_BUCKET/$trajs_dir"
    elif [ -d "$TRAJS_OH/$trajs_dir" ]; then
        echo "$TRAJS_OH/$trajs_dir"
    fi
}

push_gcs() {
    local model_name=$1
    local trajs_dir=$2
    local logs_dir=$3
    local report_file=$4
    
    local gcs_dest="$BUCKET/$model_name"
    
    echo ""
    echo "=========================================="
    echo "[GCS] Pushing: $model_name"
    echo "=========================================="
    
    # Find and push trajectories
    local trajs_path=$(find_trajs_path "$trajs_dir")
    
    if [ -n "$trajs_path" ]; then
        echo "  -> Uploading trajs from: $trajs_path"
        gsutil cp "$trajs_path/output.jsonl" "$gcs_dest/trajs/" 2>/dev/null || true
        gsutil cp "$trajs_path/output.gso.jsonl" "$gcs_dest/trajs/" 2>/dev/null || true
        gsutil cp "$trajs_path/metadata.json" "$gcs_dest/trajs/" 2>/dev/null || true
    else
        echo "  -> SKIP trajs (not found)"
    fi
    
    # Push logs
    local logs_path="$LOGS_BASE/$logs_dir"
    if [ -d "$logs_path" ]; then
        echo "  -> Uploading logs..."
        gsutil -m cp -r "$logs_path"/* "$gcs_dest/logs/" 2>&1 | grep -v "No such file or directory" || true
    else
        echo "  -> SKIP logs (not found)"
    fi
    
    # Push report
    local report_path="$REPORTS_DIR/$report_file"
    if [ -f "$report_path" ]; then
        echo "  -> Uploading report..."
        gsutil cp "$report_path" "$gcs_dest/report.json"
    else
        echo "  -> SKIP report (not found)"
    fi
    
    echo "  -> Done: $gcs_dest"
}

push_docent() {
    local model_name=$1
    local trajs_dir=$2
    local logs_dir=$3
    local report_file=$4
    
    echo ""
    echo "=========================================="
    echo "[Docent] Pushing: $model_name"
    echo "=========================================="
    
    local trajs_path=$(find_trajs_path "$trajs_dir")
    
    if [ -z "$trajs_path" ]; then
        echo "  -> SKIP (no trajectories found)"
        return
    fi
    
    local logs_path="$LOGS_BASE/$logs_dir"
    local logs_arg=""
    if [ -d "$logs_path" ]; then
        logs_arg="--logs-dir $logs_path"
    fi
    
    local report_path="$REPORTS_DIR/$report_file"
    local report_arg=""
    if [ -f "$report_path" ]; then
        report_arg="--report-file $report_path"
    fi
    
    # Run ingestion
    local output
    output=$(cd "$REPO_DIR" && uv run python scripts/docent_ingest.py \
        --submission-dir "$trajs_path" \
        --collection-name "GSO - $model_name" \
        $logs_arg $report_arg 2>&1)
    
    echo "$output" | tail -5
    
    # Extract collection ID and update models.json
    local collection_id=$(echo "$output" | grep -oP "collection: \K[a-f0-9-]+")
    if [ -n "$collection_id" ]; then
        DOCENT_COLLECTIONS[$model_name]=$collection_id
        # Update models.json with new docent_id
        cd "$REPO_DIR" && uv run python -c "
import json
with open('models.json') as f:
    models = json.load(f)
models['$model_name']['docent_id'] = '$collection_id'
with open('models.json', 'w') as f:
    json.dump(models, f, indent=2)
" 2>/dev/null
        echo "  -> Updated models.json with docent_id"
    fi
}

sync_reports() {
    echo ""
    echo "=========================================="
    echo "Syncing reports to repo..."
    echo "=========================================="
    cd "$REPO_DIR" && uv run python scripts/sync_reports.py
}

process_models() {
    local action=$1
    
    # Read models from JSON and process each (fourth column: docent_id or empty)
    local models=$(cd "$REPO_DIR" && uv run python -c "
import json
with open('models.json') as f:
    models = json.load(f)
for name, cfg in models.items():
    docent_id = cfg.get('docent_id') or ''
    print(f\"{name}|{cfg['trajs_dir']}|{cfg['logs_dir']}|{cfg['report_file']}|{docent_id}\")
")
    
    local new_only="${2:-}"
    while IFS='|' read -r model_name trajs_dir logs_dir report_file docent_id; do
        if [ "$new_only" = "new-only" ] && [ -n "$docent_id" ]; then
            echo ""
            echo "[Skip] $model_name (already has docent_id, use without new-only to re-push)"
            continue
        fi
        if [ "$action" = "gcs" ] || [ "$action" = "all" ]; then
            push_gcs "$model_name" "$trajs_dir" "$logs_dir" "$report_file"
        fi
        if [ "$action" = "docent" ] || [ "$action" = "all" ]; then
            push_docent "$model_name" "$trajs_dir" "$logs_dir" "$report_file"
        fi
    done <<< "$models"
}

# Parse args
TARGET="${1:-all}"
NEW_ONLY="${2:-}"

case "$TARGET" in
    gcs)
        echo "Pushing to GCS..."
        process_models "gcs" "$NEW_ONLY"
        sync_reports
        echo ""
        echo "Done! View at: gsutil ls $BUCKET/"
        ;;
    docent)
        echo "Pushing to Docent..."
        process_models "docent" "$NEW_ONLY"
        sync_reports
        echo ""
        echo "=========================================="
        echo "Docent Collections:"
        echo "=========================================="
        for name in "${!DOCENT_COLLECTIONS[@]}"; do
            echo "  $name: https://docent.transluce.org/dashboard/${DOCENT_COLLECTIONS[$name]}"
        done
        ;;
    all)
        echo "Pushing to GCS and Docent..."
        process_models "all" "$NEW_ONLY"
        sync_reports
        echo ""
        echo "=========================================="
        echo "Summary:"
        echo "=========================================="
        echo "GCS: gsutil ls $BUCKET/"
        echo ""
        echo "Docent Collections:"
        for name in "${!DOCENT_COLLECTIONS[@]}"; do
            echo "  $name: https://docent.transluce.org/dashboard/${DOCENT_COLLECTIONS[$name]}"
        done
        ;;
    *)
        echo "Usage: $0 [gcs|docent|all] [new-only]"
        echo "  new-only: when pushing to Docent, skip models that already have docent_id (keeps existing links)"
        exit 1
        ;;
esac
