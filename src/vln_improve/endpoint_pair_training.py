"""Read-only, train-only loading of sealed four-rollout endpoint pairs.

Data identity binds the collection identity and the ordered pair-manifest/data
byte digests. Collection resource timings are deliberately excluded because
PairStore.seal refreshes those after a complete-cache resume. SHA checks prove
consistency with the supplied records; callers can pin registered identity,
data, and asset digests to bind them to an independently approved source.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import torch

from .endpoint_pairs import ORDERS, SCHEMA, SLOTS, validate_pair as validate_pair_payload
from .endpoint_probe import COMMON_KEYS, FEATURE_DIM, FEATURE_SCHEMA
from .protocol import file_sha256, object_sha256

DATA_SCHEMA = "duet_endpoint_pair_training_data_v1"
# Source D3 protocol (configs/endpoint_pair_diagnostic.json), not the frozen
# collector's model/GPU seed. Legacy collection identity records only the latter.
SOURCE_PAIR_SELECTION_SEED = 20261003


def _sha(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"invalid {label} SHA-256")
    return value


def _number(value, *, lower=0):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= lower


def _fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} schema fields differ")


def _ordinary(path, *, directory=False):
    if path.is_symlink() or not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"missing/non-ordinary pair cache {'directory' if directory else 'file'}: {path}")


def _json(path):
    _ordinary(path)
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")))


def _committed(directory):
    manifest = _json(directory / "manifest.json")
    digest = file_sha256(directory / "manifest.json")
    if _json(directory / "COMMITTED.json") != {"manifest_sha256": digest}:
        raise ValueError("pair cache commit/manifest SHA-256 mismatch")
    return manifest, digest


def _identity(identity, split):
    _fields(identity, {"schema", "split", "usage", "selection", "selection_sha256", "seed",
                      "runtime_config_sha256", "collection_config_sha256", "coverage_report_sha256",
                      "code_files", "common_provenance", "feature_schema", "feature_dim",
                      "execution", "candidate_cache"}, "collection identity")
    if (identity["schema"] != SCHEMA or identity["split"] != split
            or identity["usage"] != ("training" if split == "train_fit" else "analysis_only")
            or identity["feature_schema"] != FEATURE_SCHEMA or identity["feature_dim"] != FEATURE_DIM
            or type(identity["seed"]) is not int or identity["seed"] < 0
            or any(not isinstance(identity[k], str) or not identity[k] for k in ("execution", "candidate_cache"))):
        raise ValueError("collection split/usage/feature/seed identity mismatch")
    for key in ("runtime_config_sha256", "collection_config_sha256", "coverage_report_sha256"):
        _sha(identity[key], key)
    common = identity["common_provenance"]
    _fields(common, COMMON_KEYS, "common provenance")
    for key in ("base_checkpoint_sha256", "feature_sha256", "annotation_sha256", "connectivity_sha256"):
        _sha(common[key], key)
    model = common["model"]
    if (not isinstance(model, dict) or model.get("dataset") != "r2r"
            or model.get("fusion") != "dynamic" or model.get("batch_size") != 1
            or model.get("max_action_len") != 15 or model.get("enc_full_graph") is not True
            or model.get("act_visited_nodes", False) is not False
            or not isinstance(common["upstream_lock"], dict) or not common["upstream_lock"]
            or type(common["partition_seed"]) is not int
            or not _number(common["dev_fraction"]) or not 0 < common["dev_fraction"] < 1
            or not isinstance(common["torch_version"], str) or not common["torch_version"]):
        raise ValueError("invalid paired-cache model/partition/runtime provenance")
    code = identity["code_files"]
    if not isinstance(code, dict) or not code:
        raise ValueError("missing paired-cache source identity")
    for name, digest in code.items():
        if (not isinstance(name, str) or not name or Path(name).is_absolute()
                or ".." in Path(name).parts or "\\" in name):
            raise ValueError("invalid paired-cache source filename")
        _sha(digest, name)


def _selection(identity, selection_seed):
    if type(selection_seed) is not int or selection_seed < 0:
        raise ValueError("expected source pair selection seed must be a nonnegative integer")
    rows = identity["selection"]
    if not isinstance(rows, list) or not rows or identity["selection_sha256"] != object_sha256(rows):
        raise ValueError("empty or inconsistent paired selection")
    names, paths, instructions = [], set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid selected pair")
        for key in ("scan", "start"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError("invalid pair scene/start")
        for key in ("path_ids", "instr_ids", "goal_vpids"):
            values = row.get(key)
            if (not isinstance(values, list) or len(values) != 2
                    or any(not isinstance(x, str) or not x for x in values) or len(set(values)) != 2):
                raise ValueError("pair must contain two distinct paths/instructions/goals")
        for path_id, instr in zip(row["path_ids"], row["instr_ids"]):
            if re.fullmatch(re.escape(path_id) + r"_[0-9]+", instr) is None:
                raise ValueError("paired instruction does not belong to its path slot")
            key = (row["scan"], path_id)
            if key in paths or instr in instructions:
                raise ValueError("paired selection repeats a path or instruction")
            paths.add(key); instructions.add(instr)
        sha = _sha(row.get("selection_hash"), "pair selection")
        if sha != object_sha256([selection_seed, row["scan"], *row["path_ids"]]):
            raise ValueError("pair selection hash does not bind its scene/path mapping")
        names.append("pair-" + sha)
        headings = row.get("heading_rad")
        if (not isinstance(headings, list) or len(headings) != 2
                or any(not _number(x, lower=-math.inf) for x in headings)
                or not _number(row.get("heading_difference_deg")) or row["heading_difference_deg"] > 1e-6
                or abs(math.degrees((headings[0] - headings[1] + math.pi) % (2 * math.pi) - math.pi)) > 1e-6
                or not _number(row.get("goal_separation_m")) or row["goal_separation_m"] <= 6):
            raise ValueError("pair heading/goal-separation rule differs")
        histories = row.get("histories")
        if not isinstance(histories, dict) or set(histories) != set(ORDERS):
            raise ValueError("both fixed history orders are required")
        vp_sets = []
        for order in ORDERS:
            history = histories[order]
            if not isinstance(history, dict):
                raise ValueError("invalid paired history metadata")
            vps = history.get("observed_vpids")
            if (not isinstance(vps, list) or not 2 <= len(vps) <= 15
                    or any(not isinstance(v, str) or not v for v in vps)
                    or len(set(vps)) != len(vps) or vps[0] != row["start"]
                    or not set(row["goal_vpids"]).issubset(vps)):
                raise ValueError("invalid paired observation history")
            walk = history.get("reference_walk")
            if (not isinstance(walk, list) or any(not isinstance(v, str) or not v for v in walk)
                    or list(dict.fromkeys(walk)) != vps or history.get("observed_states") != len(vps)
                    or history.get("reference_walk_states") != len(walk)
                    or history.get("goal_first_observation_indices") != {v: vps.index(v) for v in row["goal_vpids"]}):
                raise ValueError("paired reference/observed history mapping differs")
            first, second = row["goal_vpids"] if order == ORDERS[0] else row["goal_vpids"][::-1]
            if vps.index(first) >= vps.index(second) or vps[-1] != second:
                raise ValueError("paired endpoint observation order differs")
            vp_sets.append(set(vps))
        if vp_sets[0] != vp_sets[1]:
            raise ValueError("opposite history orders observe different node sets")
    if len(set(names)) != len(names) or names != sorted(names):
        raise ValueError("pair selection must be unique and in fixed hash order")
    return names


def _payload_contract(payload, identity_sha, pair):
    """Reuse collector parity checks and verify training-label slot semantics."""
    _fields(payload, {"schema", "identity_sha256", "pair", "feature_schema", "rollouts"}, "pair payload")
    validate_pair_payload(payload, identity_sha, pair)
    by_viewpoint, language_ids = {}, {}
    for order in ORDERS:
        vps = pair["histories"][order]["observed_vpids"]
        for index, slot in enumerate(SLOTS):
            run = payload["rollouts"][order][slot]
            _fields(run, {"instr_id", "feature_schema", "mode", "language_input_sha256", "instruction_text_sha256",
                          "features", "states", "trajectory", "forced_actions", "actual_length_m",
                          "natural_stop_probability", "wall_seconds", "instruction_slot", "labels"}, "paired rollout")
            _fields(run["labels"], {"goal_vpids", "distance_to_goals", "within_success_radius"}, "paired labels")
            if type(run["instruction_slot"]) is not int or not _number(run["actual_length_m"]) or not _number(run["wall_seconds"]):
                raise ValueError("invalid rollout slot/length/resources")
            lang = tuple(_sha(run[k], k) for k in ("language_input_sha256", "instruction_text_sha256"))
            if slot in language_ids and language_ids[slot] != lang:
                raise ValueError("same instruction changed across history orders")
            language_ids[slot] = lang
            previous, previous_length = [[vps[0]]], 0.0
            for t, state in enumerate(run["states"]):
                prefix, length = state["trajectory_prefix"], state.get("prefix_length_m")
                if (type(state["step"]) is not int or type(state.get("view_index")) is not int
                        or not 0 <= state["view_index"] < 36 or not _number(length)
                        or length < previous_length or (t == 0 and (prefix != previous or length != 0))):
                    raise ValueError("invalid paired prefix step/length/orientation")
                if t and (len(prefix) != len(previous) + 1 or prefix[:-1] != previous
                          or not isinstance(prefix[-1], list) or not prefix[-1]
                          or any(v not in vps[:t] for v in prefix[-1][:-1])):
                    raise ValueError("paired transit uses a future/unobserved node")
                previous, previous_length = prefix, length
                distance = run["labels"]["distance_to_goals"][t]
                if vps[t] in by_viewpoint and not torch.equal(by_viewpoint[vps[t]], distance):
                    raise ValueError("same viewpoint has inconsistent labels across instructions/orders")
                by_viewpoint[vps[t]] = distance
            if run["actual_length_m"] != previous_length:
                raise ValueError("paired full prefix length differs from final state")
            for j, goal in enumerate(pair["goal_vpids"]):
                cross = float(run["labels"]["distance_to_goals"][vps.index(goal), 1 - j])
                if not math.isclose(cross, pair["goal_separation_m"], abs_tol=1e-8, rel_tol=1e-10):
                    raise ValueError("goal-column labels disagree with selected goal geometry")


@dataclass(frozen=True)
class EndpointPairCache:
    root: Path
    identity: dict
    identity_sha256: str
    collection_sha256: str
    manifest: dict
    manifest_sha256: str
    data_sha256: str
    pairs: tuple[dict, ...]
    source_selection_seed: int

    @property
    def split(self):
        return self.identity["split"]

    @property
    def common_provenance(self):
        return copy.deepcopy(self.identity["common_provenance"])

    @property
    def common_identity(self):
        common = {k: copy.deepcopy(self.identity[k]) for k in
                  ("schema", "seed", "runtime_config_sha256", "collection_config_sha256", "coverage_report_sha256",
                   "code_files", "common_provenance", "feature_schema", "feature_dim", "execution", "candidate_cache")}
        return {**common, "source_pair_selection_seed": self.source_selection_seed}


@dataclass(frozen=True)
class EndpointPairExample:
    pair_id: str
    scan_id: str
    order: str
    instruction_slot: int
    instr_id: str
    features: torch.Tensor
    targets: torch.Tensor
    viewpoints: tuple[str, ...]
    goal_steps: tuple[int, int]


def pair_training_examples(payload):
    """Four complete rollout examples; targets select the instruction's own column.

    Features are only the saved 1536 STOP tokens at each visit's prefix. No goal
    distance, mask, identifier, or future endpoint is appended to model inputs.
    Call this only for payloads returned by the strict loader.
    """
    pair = payload["pair"]
    examples = []
    for order in ORDERS:
        vps = tuple(pair["histories"][order]["observed_vpids"])
        for index, slot in enumerate(SLOTS):
            run = payload["rollouts"][order][slot]
            examples.append(EndpointPairExample(pair["selection_hash"], pair["scan"], order, index,
                pair["instr_ids"][index], run["features"].detach().clone(),
                run["labels"]["within_success_radius"][:, index].float().clone(), vps,
                tuple(vps.index(v) for v in pair["goal_vpids"])))
    return tuple(examples)


def load_endpoint_pair_cache(directory, expected_split, *, expected_identity_sha256=None,
                             expected_data_sha256=None, expected_common_provenance=None,
                             expected_selection_seed=SOURCE_PAIR_SELECTION_SEED):
    """Validate a sealed train_fit/train_dev cache without modifying any file.

    ``expected_selection_seed`` binds the original CPU D3 selection protocol.
    It must not be inferred from ``identity.seed``, which is the model/GPU seed.
    The external default is the frozen D3 protocol's 20261003 selection seed.
    """
    if expected_split not in {"train_fit", "train_dev"}:
        raise ValueError("endpoint pairs allow train_fit/train_dev only")
    original = Path(directory).absolute()
    # Reject symlinks in the supplied path, not just those within the cache.
    for part in (original, *original.parents):
        if part.is_symlink():
            raise ValueError("paired cache path contains a symbolic link")
    _ordinary(original, directory=True)
    root = original.resolve(strict=True)
    collection = _json(root / "COLLECTION.json")
    _fields(collection, {"schema", "identity", "identity_sha256"}, "collection")
    identity, identity_sha = collection["identity"], collection["identity_sha256"]
    if collection["schema"] != SCHEMA or identity_sha != object_sha256(identity):
        raise ValueError("paired collection identity checksum mismatch")
    _identity(identity, expected_split)
    if expected_identity_sha256 is not None and identity_sha != _sha(expected_identity_sha256, "expected identity"):
        raise ValueError("paired collection differs from registered identity digest")
    if expected_common_provenance is not None and identity["common_provenance"] != expected_common_provenance:
        raise ValueError("paired common provenance differs from registered source")
    # Current PairStore embeds identity in COLLECTION; accept an optional exact
    # redundant identity file without requiring a format the collector never wrote.
    if (root / "IDENTITY.json").exists() or (root / "IDENTITY.json").is_symlink():
        if _json(root / "IDENTITY.json") != identity:
            raise ValueError("redundant pair IDENTITY.json mismatch")
    names = _selection(identity, expected_selection_seed)
    if {p.name for p in root.glob("pair-*")} != set(names):
        raise ValueError("paired cache directory inventory is incomplete or unexpected")
    manifest, manifest_sha = _committed(root)
    _fields(manifest, {"schema", "identity_sha256", "split", "usage", "files", "pairs", "rollouts", "states",
                      "all_shared_history_exact_parity", "resources", "navigation_metrics", "interpretation"}, "collection manifest")
    if (manifest["schema"] != SCHEMA or manifest["identity_sha256"] != identity_sha
            or manifest["split"] != expected_split or manifest["usage"] != identity["usage"]
            or manifest["all_shared_history_exact_parity"] is not True or manifest["navigation_metrics"] is not None
            or not isinstance(manifest["files"], list) or len(manifest["files"]) != len(names)):
        raise ValueError("paired collection manifest split/usage/parity/inventory mismatch")
    payloads, digests, total_states = [], [], 0
    for name, pair, entry in zip(names, identity["selection"], manifest["files"]):
        directory = root / name
        _ordinary(directory, directory=True)
        if directory.resolve().parent != root:
            raise ValueError("pair directory escapes its collection")
        if {path.name for path in directory.iterdir()} != {"manifest.json", "COMMITTED.json", "data.pt"}:
            raise ValueError("pair directory has missing or unexpected files")
        pair_manifest, pair_sha = _committed(directory)
        _fields(pair_manifest, {"schema", "identity_sha256", "pair_sha256", "data_sha256", "rollouts", "states",
                                "shared_history_exact_parity"}, "pair manifest")
        if entry != {"name": name, "manifest_sha256": pair_sha, **pair_manifest}:
            raise ValueError("ordered pair mapping/manifest bytes differ from collection")
        data_path = directory / "data.pt"
        _ordinary(data_path)
        data_sha = file_sha256(data_path)
        if (pair_manifest["schema"] != SCHEMA or pair_manifest["identity_sha256"] != identity_sha
                or pair_manifest["pair_sha256"] != object_sha256(pair)
                or pair_manifest["data_sha256"] != data_sha or pair_manifest["rollouts"] != 4
                or pair_manifest["shared_history_exact_parity"] is not True):
            raise ValueError("paired manifest/data SHA-256 or identity mismatch")
        payload = torch.load(data_path, map_location="cpu", weights_only=True)
        try:
            _payload_contract(payload, identity_sha, pair)
        except (KeyError, TypeError, IndexError, AttributeError) as exc:
            raise ValueError("malformed paired payload") from exc
        states = sum(len(run["states"]) for order in payload["rollouts"].values() for run in order.values())
        if type(pair_manifest["states"]) is not int or pair_manifest["states"] != states:
            raise ValueError("paired manifest state count mismatch")
        total_states += states
        payloads.append(payload)
        digests.append({"name": name, "manifest_sha256": pair_sha, "data_sha256": data_sha})
    expected_counts = {"pairs": len(names), "rollouts": 4 * len(names), "states": total_states}
    if any(type(manifest[k]) is not int or manifest[k] != value for k, value in expected_counts.items()):
        raise ValueError("paired collection counts differ from contents")
    data_sha = object_sha256({"schema": DATA_SCHEMA, "collection_identity_sha256": identity_sha, "pairs": digests})
    if expected_data_sha256 is not None and data_sha != _sha(expected_data_sha256, "expected data"):
        raise ValueError("paired cache differs from registered data digest")
    return EndpointPairCache(root, identity, identity_sha, file_sha256(root / "COLLECTION.json"),
                             manifest, manifest_sha, data_sha, tuple(payloads), expected_selection_seed)


def validate_endpoint_pair_splits(train, dev):
    """Require shared collection protocol/model and independent fit/dev scenes."""
    if train.split != "train_fit" or dev.split != "train_dev" or train.common_identity != dev.common_identity:
        raise ValueError("paired fit/dev split or common provenance mismatch")
    for key in ("scan", "path_ids", "instr_ids"):
        left = {value for p in train.identity["selection"] for value in
                ([p[key]] if key == "scan" else p[key])}
        right = {value for p in dev.identity["selection"] for value in
                 ([p[key]] if key == "scan" else p[key])}
        if left & right:
            raise ValueError(f"paired fit/dev share {key}")
    return train.common_identity
