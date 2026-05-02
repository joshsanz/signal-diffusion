#!/usr/bin/env python3
"""Generate sky-infer-*.yaml configs from finished wandb training runs.

For each finished wandb run in the given time window, reads best_eval/metadata.json
from R2 via rclone to find the best checkpoint, then writes a ready-to-use
sky-infer-*.yaml alongside the existing sky-training-* configs.
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import wandb
import yaml

# ── Literal block scalar support for PyYAML ──────────────────────────────────

class _Literal(str):
    pass


def _literal_representer(dumper: yaml.Dumper, data: "_Literal") -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(_Literal, _literal_representer)


# ── Variant registry ──────────────────────────────────────────────────────────

@dataclass
class VariantSpec:
    training_config: str    # filename under config/
    diffusion_toml: str     # filename under config/diffusion/
    model_family: str       # "hourglass" or "sd35"
    batch_size: int
    output_dir_template: str  # /data/runs/... with {timestamp} placeholder


# Maps the logging.run_name stored in wandb → everything needed to build the infer config.
# output_dir_template matches the --output_dir arg in each sky-training-*.yaml run section.
VARIANTS: dict[str, VariantSpec] = {
    "hourglass-db-only": VariantSpec(
        "sky-training-hrgls.yaml", "hourglass-db-only.toml", "hourglass", 256,
        "/data/runs/hourglass/db-only-{timestamp}",
    ),
    "hourglass-db-iq": VariantSpec(
        "sky-training-hrgls-dbiq.yaml", "hourglass-db-iq.toml", "hourglass", 256,
        "/data/runs/hourglass/db-iq-{timestamp}",
    ),
    "hourglass-db-polar": VariantSpec(
        "sky-training-hrgls-dbpolar.yaml", "hourglass-db-polar.toml", "hourglass", 256,
        "/data/runs/hourglass/db-polar-{timestamp}",
    ),
    "hourglass-timeseries": VariantSpec(
        "sky-training-hrgls-timeseries.yaml", "hourglass-timeseries.toml", "hourglass", 256,
        "/data/runs/hourglass/timeseries-{timestamp}",
    ),
    "sd35-db-only": VariantSpec(
        "sky-training-sd35.yaml", "sd35-db-only.toml", "sd35", 128,
        "/data/runs/sd35/db-only-{timestamp}",
    ),
    "sd35-db-iq": VariantSpec(
        "sky-training-sd35-dbiq.yaml", "sd35-db-iq.toml", "sd35", 128,
        "/data/runs/sd35/db-iq-{timestamp}",
    ),
    "sd35-db-polar": VariantSpec(
        "sky-training-sd35-dbpolar.yaml", "sd35-db-polar.toml", "sd35", 128,
        "/data/runs/sd35/db-polar-{timestamp}",
    ),
}

_TIMESTAMP_RE = re.compile(r"^(.+)-(\d{8}_\d{6})$")


# ── R2 helpers ────────────────────────────────────────────────────────────────

def _rclone_cat(remote: str, bucket: str, path: str) -> str | None:
    """Return the text content of an R2 file, or None if not found."""
    result = subprocess.run(
        ["rclone", "cat", f"{remote}:{bucket}/{path}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _rclone_lsf_dirs(remote: str, bucket: str, path: str) -> list[str]:
    """Return directory names inside an R2 prefix (no trailing slash)."""
    result = subprocess.run(
        ["rclone", "lsf", "--dirs-only", f"{remote}:{bucket}/{path}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [d.rstrip("/") for d in result.stdout.splitlines() if d.strip()]


def find_best_checkpoint_path(
    output_dir: str, run, r2_remote: str, r2_bucket: str
) -> str | None:
    """Return the full /data/... path to the best checkpoint dir, or None.

    Strategy:
    1. List available checkpoint steps from best_eval/ or checkpoints/ on R2.
    2. Query wandb run history for eval/kid_mean (or eval/loss) at those steps.
    3. Pick the step with the lowest metric value.
    4. Fall back to best_eval/metadata.json, then highest available step.
    """
    r2_rel = output_dir.lstrip("/")
    if r2_rel.startswith("data/"):
        r2_rel = r2_rel[len("data/"):]

    best_eval_steps = _parse_checkpoint_steps(
        _rclone_lsf_dirs(r2_remote, r2_bucket, f"{r2_rel}/best_eval")
    )
    checkpoints_steps = _parse_checkpoint_steps(
        _rclone_lsf_dirs(r2_remote, r2_bucket, f"{r2_rel}/checkpoints")
    )

    if best_eval_steps:
        base, available = "best_eval", set(best_eval_steps)
    elif checkpoints_steps:
        base, available = "checkpoints", set(checkpoints_steps)
    else:
        return None

    best_step = _best_step_from_wandb(run, available)

    if best_step is None and base == "best_eval":
        # Fall back to metadata.json written by the trainer
        content = _rclone_cat(r2_remote, r2_bucket, f"{r2_rel}/best_eval/metadata.json")
        if content:
            try:
                entries = json.loads(content)
                if entries:
                    best_step = int(sorted(entries, key=lambda x: x["kid_mean"])[0]["step"])
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

    if best_step is None:
        best_step = max(available)

    return f"{output_dir}/{base}/checkpoint-{best_step}"


def _parse_checkpoint_steps(dirs: list[str]) -> list[int]:
    steps = []
    for d in dirs:
        m = re.match(r"checkpoint-(\d+)$", d)
        if m:
            steps.append(int(m.group(1)))
    return steps


def _best_step_from_wandb(run, available_steps: set[int]) -> int | None:
    """Return the available step with the lowest eval/kid_mean, or eval/loss as fallback.

    Metrics may not be logged at exactly the checkpoint steps, so for each checkpoint
    step we find the nearest logged value and pick the checkpoint with the lowest one.
    """
    for metric in ("eval/kid_mean", "eval/loss", "val/loss"):
        try:
            rows = run.history(keys=[metric], pandas=False, samples=100_000)
        except Exception:
            continue
        metric_at_step = [
            (float(row[metric]), int(row["_step"]))
            for row in rows
            if row.get(metric) is not None
        ]
        if not metric_at_step:
            continue
        # For each checkpoint step find the nearest logged metric value, then pick lowest
        candidates = []
        for ckpt_step in available_steps:
            val, _ = min(metric_at_step, key=lambda x: abs(x[1] - ckpt_step))
            candidates.append((val, ckpt_step))
        best_val, best_step = min(candidates)
        print(f"{metric}={best_val:.4f}@step={best_step}", end=" ")
        return best_step
    return None


# ── Config generation ─────────────────────────────────────────────────────────

def _build_run_section(
    spec: VariantSpec,
    run_name_prefix: str,
    run_timestamp: str,
    checkpoint_path: str,
    n_samples: int,
) -> str:
    lines = [
        "echo \"Hello, SkyPilot!\"",
        f"export RUN_ID={run_timestamp}",
        f"export RUN_PATH={checkpoint_path}",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "source .venv/bin/activate",
        "./scripts/prep-configs-for-sky.sh",
    ]
    if spec.model_family == "sd35":
        lines.append(
            f"python scripts/edit_config.py -c config/diffusion/{spec.diffusion_toml}"
            " -s training.compile_model=false"
        )
    lines += [
        f"python scripts/generate_synthetic_dataset.py"
        f" -m $RUN_PATH"
        f" -c config/diffusion/{spec.diffusion_toml}"
        f" -o /data/generated/{run_name_prefix}-$RUN_ID/"
        f" -n {n_samples}"
        f" -b {spec.batch_size}"
        " --overwrite",
        "echo \"Run completed.\"",
    ]
    return "\n".join(lines) + "\n"


def generate_infer_config(
    spec: VariantSpec,
    run_name_prefix: str,
    run_timestamp: str,
    checkpoint_path: str,
    n_samples: int,
    config_dir: Path,
) -> dict:
    """Build the infer config dict by cloning the training template and swapping run."""
    training_path = config_dir / spec.training_config
    with open(training_path) as f:
        data = yaml.safe_load(f)

    run_script = _build_run_section(
        spec, run_name_prefix, run_timestamp, checkpoint_path, n_samples
    )
    data["run"] = _Literal(run_script)

    # Wrap setup in a literal block scalar too (preserves shell formatting)
    if "setup" in data and isinstance(data["setup"], str):
        data["setup"] = _Literal(data["setup"])

    return data


# ── wandb helpers ─────────────────────────────────────────────────────────────

def _output_dir_from_metadata(run) -> str | None:
    """Extract --output_dir from the args list captured in run.metadata."""
    try:
        args = run.metadata.get("args", [])
        for i, arg in enumerate(args):
            if arg in ("--output_dir", "--output-dir") and i + 1 < len(args):
                return args[i + 1]
    except Exception:
        pass
    return None

def _parse_datetime(s: str) -> datetime:
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Cannot parse date '{s}'. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS."
    )


def query_wandb_runs(
    project: str,
    entity: str | None,
    since: datetime,
    until: datetime,
) -> list:
    api = wandb.Api()
    resolved_entity = entity or api.default_entity
    since_iso = since.strftime("%Y-%m-%dT%H:%M:%S")
    until_iso = until.strftime("%Y-%m-%dT%H:%M:%S")
    runs = api.runs(
        f"{resolved_entity}/{project}",
        filters={
            "created_at": {"$gte": since_iso, "$lte": until_iso},
            "state": "finished",
        },
    )
    return list(runs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sky-infer-*.yaml configs from finished wandb runs."
    )
    parser.add_argument("-s", "--since", required=True, type=_parse_datetime,
                        metavar="DATE", help="Start of time window (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
    parser.add_argument("-u", "--until", type=_parse_datetime, default=None,
                        metavar="DATE", help="End of time window (default: now)")
    parser.add_argument("--project", default="signal-diffusion",
                        help="wandb project name (default: signal-diffusion)")
    parser.add_argument("--entity", default=None,
                        help="wandb entity (default: auto-detect from login)")
    parser.add_argument("--output-dir", type=Path, default=Path("config"),
                        metavar="DIR", help="Directory to write generated configs (default: config/)")
    parser.add_argument("--r2-remote", default="r2",
                        help="rclone remote name for R2 (default: r2)")
    parser.add_argument("--r2-bucket", default="signal-diffusion",
                        help="R2 bucket name (default: signal-diffusion)")
    parser.add_argument("-n", "--n-samples", type=int, default=50000,
                        help="Number of samples to generate (default: 50000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be created without writing files")
    args = parser.parse_args()

    until = args.until or datetime.now(tz=timezone.utc)

    print(f"Querying wandb project '{args.project}' from {args.since.date()} to {until.date()} …")
    runs = query_wandb_runs(args.project, args.entity, args.since, until)
    print(f"Found {len(runs)} finished run(s).")

    config_dir = args.output_dir
    generated = 0
    skipped = 0

    for run in runs:
        name = run.name or ""
        m = _TIMESTAMP_RE.match(name)
        if not m:
            print(f"  SKIP {name!r}: run name doesn't match expected format")
            skipped += 1
            continue

        run_name_prefix, run_timestamp = m.group(1), m.group(2)

        if run_name_prefix not in VARIANTS:
            print(f"  SKIP {name!r}: unknown variant prefix {run_name_prefix!r}")
            skipped += 1
            continue

        spec = VARIANTS[run_name_prefix]

        # Prefer the actual --output_dir arg captured in wandb metadata (most accurate).
        # Fall back to reconstructing from the variant template + run name timestamp,
        # which may be off by a few minutes due to startup time.
        output_dir = (
            run.config.get("training.output_dir")
            or _output_dir_from_metadata(run)
            or spec.output_dir_template.format(timestamp=run_timestamp)
        )

        print(f"  Processing {name!r} …", end=" ", flush=True)
        checkpoint_dir = find_best_checkpoint_path(output_dir, run, args.r2_remote, args.r2_bucket)
        if checkpoint_dir is None:
            print("SKIP (no checkpoint found on R2)")
            skipped += 1
            continue

        if spec.model_family == "sd35":
            checkpoint_path = f"{checkpoint_dir}/ema/transformer"
        else:
            checkpoint_path = f"{checkpoint_dir}/"

        out_filename = f"sky-infer-{run_name_prefix}-{run_timestamp}.yaml"
        out_path = config_dir / out_filename

        if args.dry_run:
            print(f"DRY RUN → would write {out_path}")
            print(f"    checkpoint: {checkpoint_path}")
            generated += 1
            continue

        config_data = generate_infer_config(
            spec, run_name_prefix, run_timestamp, checkpoint_path,
            args.n_samples, config_dir,
        )

        with open(out_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        print(f"wrote {out_path}")
        generated += 1

    print(f"\nDone: {generated} config(s) generated, {skipped} skipped.")


if __name__ == "__main__":
    main()
