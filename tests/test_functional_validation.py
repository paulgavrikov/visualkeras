import pytest

from visualkeras.functional import _validate_and_normalize_collapse_rules


class DummyLayer:
    pass


def test_validate_collapse_rules_none():
    assert _validate_and_normalize_collapse_rules(None) == []


def test_validate_collapse_rules_layer_rule_defaults():
    rules = _validate_and_normalize_collapse_rules(
        [
            {
                "kind": "layer",
                "selector": "block_conv",
                "repeat_count": 3,
            }
        ]
    )
    assert rules == [
        {
            "kind": "layer",
            "selector": "block_conv",
            "repeat_count": 3,
            "label": "3x",
            "annotation_position": "above",
        }
    ]


def test_validate_collapse_rules_block_rule_explicit_values():
    rules = _validate_and_normalize_collapse_rules(
        [
            {
                "kind": "block",
                "selector": ["conv", DummyLayer],
                "repeat_count": 5,
                "label": "Conv Block",
                "annotation_position": "below",
            }
        ]
    )
    assert rules[0]["kind"] == "block"
    assert rules[0]["selector"] == ("conv", DummyLayer)
    assert rules[0]["repeat_count"] == 5
    assert rules[0]["label"] == "Conv Block"
    assert rules[0]["annotation_position"] == "below"


@pytest.mark.parametrize(
    "value",
    ["not-a-sequence", 123, {"kind": "layer", "selector": "x", "repeat_count": 2}],
)
def test_validate_collapse_rules_invalid_container(value):
    with pytest.raises(TypeError):
        _validate_and_normalize_collapse_rules(value)


def test_validate_collapse_rules_rule_must_be_mapping():
    with pytest.raises(TypeError):
        _validate_and_normalize_collapse_rules([1])


def test_validate_collapse_rules_missing_selector():
    with pytest.raises(ValueError):
        _validate_and_normalize_collapse_rules([{"kind": "layer", "repeat_count": 2}])


def test_validate_collapse_rules_invalid_kind():
    with pytest.raises(ValueError):
        _validate_and_normalize_collapse_rules(
            [{"kind": "unsupported", "selector": "conv", "repeat_count": 2}]
        )


def test_validate_collapse_rules_invalid_layer_selector_type():
    with pytest.raises(TypeError):
        _validate_and_normalize_collapse_rules(
            [{"kind": "layer", "selector": ["conv"], "repeat_count": 2}]
        )


def test_validate_collapse_rules_invalid_block_selector_type():
    with pytest.raises(TypeError):
        _validate_and_normalize_collapse_rules(
            [{"kind": "block", "selector": "conv", "repeat_count": 2}]
        )


def test_validate_collapse_rules_block_selector_too_short():
    with pytest.raises(ValueError):
        _validate_and_normalize_collapse_rules(
            [{"kind": "block", "selector": ["conv"], "repeat_count": 2}]
        )


@pytest.mark.parametrize("repeat_count", [None, 0, 1, "2"])
def test_validate_collapse_rules_invalid_repeat_count(repeat_count):
    with pytest.raises(ValueError):
        _validate_and_normalize_collapse_rules(
            [{"kind": "layer", "selector": "conv", "repeat_count": repeat_count}]
        )


def test_validate_collapse_rules_invalid_annotation_position():
    with pytest.raises(ValueError):
        _validate_and_normalize_collapse_rules(
            [
                {
                    "kind": "layer",
                    "selector": "conv",
                    "repeat_count": 3,
                    "annotation_position": "left",
                }
            ]
        )


def test_validate_collapse_rules_invalid_label_type():
    with pytest.raises(TypeError):
        _validate_and_normalize_collapse_rules(
            [{"kind": "layer", "selector": "conv", "repeat_count": 3, "label": 999}]
        )
