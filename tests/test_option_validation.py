from types import MappingProxyType

import pytest
import visualkeras


@pytest.mark.parametrize("mode", ["layered", "graph", "functional", "lenet"])
@pytest.mark.parametrize("via_show", [False, True])
def test_unknown_options_fail_before_model_access_or_file_output(tmp_path, mode, via_show):
    target = tmp_path / "must-not-exist.png"
    options = MappingProxyType({"paddng": 12, "to_file": str(target)})
    renderer = visualkeras.show if via_show else getattr(visualkeras, mode + "_view")
    kwargs = {"mode": mode} if via_show else {}
    with pytest.raises(ValueError, match="Unknown .* option.*paddng.*padding"):
        renderer(object(), options=options, **kwargs)
    assert not target.exists()
    assert options["paddng"] == 12


@pytest.mark.parametrize("mode", ["layered", "graph", "functional", "lenet"])
@pytest.mark.parametrize("key", [1, "model", "options", "preset"])
def test_reserved_and_non_string_mapping_keys_are_rejected(mode, key):
    with pytest.raises(ValueError, match="Unknown .* option"):
        getattr(visualkeras, mode + "_view")(object(), options={key: None})
