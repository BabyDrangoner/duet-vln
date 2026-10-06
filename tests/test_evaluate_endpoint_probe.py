import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_endpoint_probe import validate_registration
from test_study_ledger import study, request
from vln_improve.protocol import file_sha256
from vln_improve.study_ledger import StudyLedger


def test_endpoint_eval_requires_matching_pending_full_access(tmp_path):
    config, head, study_path = (tmp_path / n for n in ("experiment.json", "head.pt", "study.json"))
    config.write_text("{}"); head.write_bytes(b"frozen model"); study_path.write_text(json.dumps(study()))
    ledger = StudyLedger(tmp_path / "ledger.jsonl", study())
    args = SimpleNamespace(split="val_unseen", head=head, limit=None, access_id="P1",
                           ledger=ledger.path, study=study_path, experiment=config, seed=0)
    report = {"episodes": [{}] * 2349}
    ledger.register(**request(config_sha256=file_sha256(config), checkpoint_sha256=file_sha256(head)))
    validate_registration(args, report, "c" * 64)
    with pytest.raises(ValueError, match="protocol differs"):
        validate_registration(args, report, "d" * 64)
    changed = copy.copy(args); changed.limit = 10
    with pytest.raises(ValueError, match="full split"):
        validate_registration(changed, report, "c" * 64)
    ledger.finish("P1", status="failed", metrics={}, resources={"seconds": None}, decision="retry requires new access", error="test")
    with pytest.raises(ValueError, match="cannot be rerun"):
        validate_registration(args, report, "c" * 64)
