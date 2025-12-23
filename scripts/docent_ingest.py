#!/usr/bin/env python3
"""
Ingest GSO trajectories (OpenHands format) into Docent.

This script adapts the SWE-bench ingestion approach for GSO's OpenHands event-based
trajectory format, including GSO-specific scoring metrics (opt_base, opt_commit, etc.).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from dotenv import load_dotenv

from docent import Docent
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import (
    parse_chat_message,
    AssistantMessage,
    UserMessage,
    ToolMessage,
)
from docent.data_models.chat.tool import ToolCall, ToolCallContent

load_dotenv(Path(__file__).parent.parent / ".env")


def find_trajectory_files(base_dir: Path) -> List[Path]:
    """Find all output.jsonl trajectory files under the base directory."""
    # Look for output.jsonl files which contain full trajectories
    return sorted(base_dir.glob("**/output.jsonl"))


def convert_openhands_history_to_messages(history: List[dict]) -> List:
    """
    Convert OpenHands event history to Docent message format.
    
    OpenHands format:
    - Events have: source (agent/user/environment), action, observation, args, content
    - Actions: run, read, write, message, think, finish, etc.
    - Observations: run output, file contents, etc.
    """
    messages = []
    pending_tool_call = None
    call_counter = 1
    
    for event in history:
        source = event.get("source", "")
        action = event.get("action")
        observation = event.get("observation")
        args = event.get("args", {})
        content = event.get("content") or event.get("message", "")
        
        # Skip system events
        if action == "system":
            continue
            
        # User messages
        if source == "user" and action == "message":
            msg_content = args.get("content", "") or content
            if msg_content:
                messages.append(UserMessage(content=msg_content))
            continue
        
        # Agent actions (tool calls)
        if source == "agent" and action and action != "message":
            thought = args.get("thought", "") or ""
            
            if action == "run":
                # Bash command
                command = args.get("command", "")
                call_id = f"call_{call_counter}"
                call_counter += 1
                messages.append(
                    AssistantMessage(
                        content=thought,
                        tool_calls=[
                            ToolCall(
                                id=call_id,
                                function="bash",
                                arguments={"command": command},
                                view=ToolCallContent(
                                    format="markdown",
                                    content=f"```bash\n{command}\n```"
                                )
                            )
                        ],
                    )
                )
                pending_tool_call = ("bash", call_id)
                
            elif action == "read":
                # File read
                path = args.get("path", "")
                call_id = f"call_{call_counter}"
                call_counter += 1
                messages.append(
                    AssistantMessage(
                        content=thought,
                        tool_calls=[
                            ToolCall(
                                id=call_id,
                                function="read_file",
                                arguments={"path": path},
                                view=ToolCallContent(
                                    format="markdown",
                                    content=f"Reading file: `{path}`"
                                )
                            )
                        ],
                    )
                )
                pending_tool_call = ("read_file", call_id)
                
            elif action == "write":
                # File write
                path = args.get("path", "")
                file_content = args.get("content", "")[:500] + "..." if len(args.get("content", "")) > 500 else args.get("content", "")
                call_id = f"call_{call_counter}"
                call_counter += 1
                messages.append(
                    AssistantMessage(
                        content=thought,
                        tool_calls=[
                            ToolCall(
                                id=call_id,
                                function="write_file",
                                arguments={"path": path},
                                view=ToolCallContent(
                                    format="markdown",
                                    content=f"Writing to file: `{path}`\n```\n{file_content}\n```"
                                )
                            )
                        ],
                    )
                )
                pending_tool_call = ("write_file", call_id)
                
            elif action == "think":
                # Thinking/reasoning
                messages.append(AssistantMessage(content=f"**Thinking:** {thought}"))
                pending_tool_call = None
                
            elif action == "finish":
                # Finish action
                final_thought = args.get("thought", "") or thought
                messages.append(AssistantMessage(content=f"**Finished:** {final_thought}"))
                pending_tool_call = None
                
            else:
                # Other actions - generic tool call
                call_id = f"call_{call_counter}"
                call_counter += 1
                messages.append(
                    AssistantMessage(
                        content=thought,
                        tool_calls=[
                            ToolCall(
                                id=call_id,
                                function=action,
                                arguments=args,
                                view=ToolCallContent(
                                    format="markdown",
                                    content=f"Action: {action}"
                                )
                            )
                        ],
                    )
                )
                pending_tool_call = (action, call_id)
            continue
        
        # Agent message (no action, just text)
        if source == "agent" and action == "message":
            msg_content = args.get("content", "") or content
            if msg_content:
                messages.append(AssistantMessage(content=msg_content))
            pending_tool_call = None
            continue
        
        # Observations (tool results)
        if observation and pending_tool_call:
            func_name, call_id = pending_tool_call
            
            # Get observation content
            obs_content = event.get("content", "")
            if not obs_content and "extras" in event:
                obs_content = str(event.get("extras", {}))
            
            # Truncate very long outputs
            if len(obs_content) > 5000:
                obs_content = obs_content[:5000] + "\n... (truncated)"
            
            messages.append(
                ToolMessage(
                    content=obs_content,
                    tool_call_id=call_id,
                    function=func_name
                )
            )
            pending_tool_call = None
            continue
    
    return messages


def load_gso_report(logs_dir: Path, instance_id: str) -> dict:
    """Load GSO evaluation report for an instance."""
    report_path = logs_dir / instance_id / "report.json"
    if not report_path.exists():
        return {}
    
    try:
        with open(report_path) as f:
            report = json.load(f)
            return report.get(instance_id, {})
    except Exception:
        return {}


def build_agent_run(
    traj_data: dict,
    logs_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    run_report: Optional[dict] = None,
) -> Optional[AgentRun]:
    """Build a Docent AgentRun from a GSO trajectory."""
    
    instance_id = traj_data.get("instance_id")
    if not instance_id:
        return None
    
    history = traj_data.get("history", [])
    if not history:
        return None
    
    # Convert OpenHands history to Docent messages
    messages = convert_openhands_history_to_messages(history)
    if not messages:
        return None
    
    transcript = Transcript(messages=messages)
    
    # Build metadata
    metadata = {
        "instance_id": instance_id,
    }
    
    # Add trajectory metadata
    traj_metadata = traj_data.get("metadata", {})
    if traj_metadata:
        metadata["agent_class"] = traj_metadata.get("agent_class")
        llm_config = traj_metadata.get("llm_config", {})
        metadata["llm_model"] = llm_config.get("model")
    
    # Add metrics
    metrics = traj_data.get("metrics", {})
    if metrics:
        metadata["metrics"] = metrics
    
    # Add instance info
    instance_info = traj_data.get("instance", {})
    if instance_info:
        metadata["repo"] = instance_info.get("repo")
        metadata["api"] = instance_info.get("api")
    
    # Add test result info
    test_result = traj_data.get("test_result", {})
    if test_result:
        metadata["has_patch"] = bool(test_result.get("git_patch"))
    
    # Load GSO-specific scoring from evaluation report
    scores = {"status": "unknown"}
    
    if logs_dir:
        instance_report = load_gso_report(logs_dir, instance_id)
        if instance_report:
            scores = {
                "test_passed": instance_report.get("test_passed", False),
                "opt_base": instance_report.get("opt_base", False),
                "opt_commit": instance_report.get("opt_commit", False),
                "opt_main": instance_report.get("opt_main", False),
                "patch_applied": instance_report.get("patch_successfully_applied", False),
            }
            
            # Add optimization stats if available
            opt_stats = instance_report.get("opt_stats", {})
            if opt_stats:
                scores["gm_speedup_patch_base"] = opt_stats.get("gm_speedup_patch_base")
                scores["gm_speedup_patch_commit"] = opt_stats.get("gm_speedup_patch_commit")
    
    # Check run-level report for instance status
    if run_report:
        instance_sets = run_report.get("instance_sets", {})
        if instance_id in instance_sets.get("passed_ids", []):
            scores["status"] = "passed"
        elif instance_id in instance_sets.get("opt_base_ids", []):
            scores["status"] = "opt_base"
        elif instance_id in instance_sets.get("test_failed_ids", []):
            scores["status"] = "test_failed"
        elif instance_id in instance_sets.get("patch_failed_ids", []):
            scores["status"] = "patch_failed"
        elif instance_id in instance_sets.get("error_ids", []):
            scores["status"] = "error"
    
    metadata["scores"] = scores
    
    if model_name:
        metadata["model_name"] = model_name
    
    # Remove None values
    metadata = {k: v for k, v in metadata.items() if v is not None}
    
    return AgentRun(transcripts=[transcript], metadata=metadata)


def load_run_report(submission_dir: Path) -> Optional[dict]:
    """Load the run-level report file."""
    # Look for report file in logs directory
    logs_dir = submission_dir / "logs"
    if not logs_dir.exists():
        return None
    
    # Find report.json file (format: model_name.run_id.report.json)
    report_files = list(logs_dir.glob("*.report.json"))
    if report_files:
        with open(report_files[0]) as f:
            return json.load(f)
    
    return None


def ingest_trajectories(
    traj_file: Path,
    collection_name: str,
    batch_size: int,
    logs_dir: Optional[Path] = None,
    existing_collection_id: Optional[str] = None,
    model_name: Optional[str] = None,
    run_report: Optional[dict] = None,
) -> str:
    """Create/update a Docent collection and upload AgentRuns in batches."""
    
    api_key = os.getenv("DOCENT_API_KEY")
    if not api_key:
        print("Error: DOCENT_API_KEY not found in .env file")
        sys.exit(1)
    
    client = Docent(api_key=api_key)
    
    # Create or use existing collection
    if existing_collection_id:
        collection_id = existing_collection_id
    else:
        collection_id = client.create_collection(
            name=collection_name,
            description="GSO benchmark trajectories"
        )
        # Make collection publicly viewable
        client.make_collection_public(collection_id)
        print(f"Created public collection: {collection_name} ({collection_id})")
    
    # Load run-level report if not provided
    if run_report is None:
        run_report = load_run_report(traj_file.parent)
    
    # Parse trajectories
    agent_runs: List[AgentRun] = []
    
    with open(traj_file) as f:
        lines = f.readlines()
    
    for line in tqdm(lines, desc="Parsing trajectories"):
        line = line.strip()
        if not line:
            continue
        
        try:
            traj_data = json.loads(line)
            run = build_agent_run(
                traj_data,
                logs_dir=logs_dir,
                model_name=model_name,
                run_report=run_report,
            )
            if run:
                agent_runs.append(run)
        except Exception as e:
            print(f"Error parsing trajectory: {e}")
            continue
    
    print(f"Prepared {len(agent_runs)} runs")
    
    # Upload in batches
    for i in tqdm(range(0, len(agent_runs), batch_size), desc="Uploading"):
        try:
            client.add_agent_runs(collection_id, agent_runs[i : i + batch_size])
        except Exception as e:
            print(f"Error uploading batch starting at index {i}: {e}")
            continue
    
    return collection_id


def run_ingestion(
    submission_dir: Path,
    collection_name: str,
    batch_size: int = 50,
    collection_id: Optional[str] = None,
    logs_dir: Optional[Path] = None,
    report_file: Optional[Path] = None,
) -> str:
    """
    Main ingestion entrypoint.
    
    Args:
        submission_dir: Directory containing output.jsonl
        collection_name: Name for the Docent collection
        batch_size: Batch size for uploads
        collection_id: Optional existing collection ID to add to
        logs_dir: Optional directory containing GSO evaluation logs
        report_file: Optional path to report.json file
    
    Returns:
        Collection ID
    """
    if not submission_dir.exists():
        raise FileNotFoundError(f"Submission directory not found: {submission_dir}")
    
    traj_file = submission_dir / "output.jsonl"
    if not traj_file.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_file}")
    
    # Check for logs directory (use provided or look in submission dir)
    if logs_dir is None:
        logs_dir = submission_dir / "logs"
    if logs_dir and not logs_dir.exists():
        logs_dir = None
        print("Note: No logs directory found, scoring info will be limited")
    
    # Load report if provided
    run_report = None
    if report_file and report_file.exists():
        try:
            with open(report_file) as f:
                run_report = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load report: {e}")
    
    # Extract model name from directory
    model_name = submission_dir.name
    
    return ingest_trajectories(
        traj_file=traj_file,
        collection_name=collection_name,
        batch_size=batch_size,
        logs_dir=logs_dir,
        existing_collection_id=collection_id,
        model_name=model_name,
        run_report=run_report,
    )


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Ingest GSO trajectories into Docent"
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        required=True,
        help="Directory containing output.jsonl trajectory file",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        required=True,
        help="Name for the Docent collection",
    )
    parser.add_argument(
        "--collection-id",
        type=str,
        default=None,
        help="Existing collection ID to add runs to",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for uploads (default: 50)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Directory containing GSO evaluation logs (optional)",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Path to report.json file (optional)",
    )
    
    args = parser.parse_args()
    
    collection_id = run_ingestion(
        submission_dir=args.submission_dir,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
        collection_id=args.collection_id,
        logs_dir=args.logs_dir,
        report_file=args.report_file,
    )
    
    if collection_id:
        print(f"\nSuccessfully ingested to collection: {collection_id}")
    else:
        print("No trajectories found")
        sys.exit(1)


if __name__ == "__main__":
    main()

