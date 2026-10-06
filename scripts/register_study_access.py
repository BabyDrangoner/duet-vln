"""Register validation access before running it, then append its outcome."""
import argparse
import json
from pathlib import Path

from vln_improve.protocol import file_sha256
from vln_improve.study_ledger import CATEGORIES, StudyLedger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("configs/research_study.json"))
    parser.add_argument("--ledger", type=Path, help="default: study protocol access_ledger_target")
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    register.add_argument("--access-id", required=True)
    register.add_argument("--category", choices=CATEGORIES, required=True)
    register.add_argument("--variant-id", required=True)
    register.add_argument("--config", type=Path, required=True, help="fixed method settings, excluding seed/checkpoint")
    register.add_argument("--checkpoint-sha256", required=True)
    register.add_argument("--code-sha256", required=True)
    register.add_argument("--purpose", required=True)
    register.add_argument("--split", default="val_unseen", choices=["val_unseen"])
    register.add_argument("--seed", type=int, required=True)
    register.add_argument("--expected-episodes", type=int, required=True)
    register.add_argument("--subset-ids", type=Path, help="JSON list of the exact fixed instruction IDs")
    register.add_argument("--label-use", choices=("evaluation", "analysis"), default="evaluation")
    register.add_argument("--budget", type=Path)
    for command in ("complete", "fail"):
        finish = commands.add_parser(command)
        finish.add_argument("--access-id", required=True)
        finish.add_argument("--metrics", type=Path, required=command == "complete", help="JSON object of numeric metrics")
        finish.add_argument("--resources", type=Path, required=True, help="JSON object of measured costs or explicit nulls")
        finish.add_argument("--report", type=Path, required=command == "complete")
        finish.add_argument("--decision", required=True)
        if command == "fail":
            finish.add_argument("--error", required=True)
    commands.add_parser("status")
    args = parser.parse_args()
    study = json.loads(args.study.read_text())
    ledger = StudyLedger(args.ledger or Path(study["evaluation_protocol"]["access_ledger_target"]), study)
    if args.command == "register":
        result = ledger.register(
            access_id=args.access_id, category=args.category, variant_id=args.variant_id,
            config_sha256=file_sha256(args.config), checkpoint_sha256=args.checkpoint_sha256,
            code_sha256=args.code_sha256, purpose=args.purpose, split=args.split, seed=args.seed,
            expected_episodes=args.expected_episodes, subset=args.subset_ids is not None,
            subset_ids=json.loads(args.subset_ids.read_text()) if args.subset_ids else None,
            label_use=args.label_use, budget_path=args.budget)
    elif args.command in {"complete", "fail"}:
        result = ledger.finish(args.access_id, status="completed" if args.command == "complete" else "failed",
                               metrics=json.loads(args.metrics.read_text()) if args.metrics else {},
                               resources=json.loads(args.resources.read_text()), decision=args.decision,
                               report_path=args.report, error=getattr(args, "error", None))
    else:
        result = None
    print(json.dumps(dict(record=result, status=ledger.status()), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
