from vln_improve.protocol import select_partition, object_sha256


def test_training_and_development_never_share_scenes():
    records = [{"scan": f"scene{i}", "instr_id": f"{i}_{j}"} for i in range(12) for j in range(3)]
    fit = select_partition(records, "train_fit")
    dev = select_partition(records, "train_dev")
    assert not {x["scan"] for x in fit} & {x["scan"] for x in dev}
    assert len(fit) + len(dev) == len(records)
    assert {x["scan"] for x in fit} == {x["scan"] for x in select_partition(list(reversed(records)), "train_fit")}


def test_protocol_hash_ignores_dictionary_order_but_detects_changes():
    assert object_sha256({"a": 1, "b": 2}) == object_sha256({"b": 2, "a": 1})
    assert object_sha256({"a": 1}) != object_sha256({"a": 2})
