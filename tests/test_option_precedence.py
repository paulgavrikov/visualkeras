import pytest
import visualkeras
from PIL import Image

tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.integration


def _build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8, 8, 3), name="option_input"),
        tf.keras.layers.Conv2D(4, 3, name="option_conv"),
        tf.keras.layers.Flatten(name="option_flatten"),
        tf.keras.layers.Dense(3, name="option_output"),
    ])


def _render(mode, model, via_show, **kwargs):
    if via_show:
        return visualkeras.show(model, mode=mode, **kwargs)
    return getattr(visualkeras, mode + "_view")(model, **kwargs)


@pytest.mark.parametrize("via_show", [False, True])
def test_layered_explicit_default_spacing_overrides_preset(via_show):
    model = _build_model()
    image = _render("layered", model, via_show, preset="compact", spacing=10, legend_text_spacing_offset=0)
    reference = visualkeras.layered_view(model, spacing=10, padding=6, legend_text_spacing_offset=0)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("via_show", [False, True])
def test_layered_explicit_none_resets_option_caption(via_show):
    model = _build_model()
    image = _render(
        "layered", model, via_show,
        options=visualkeras.LayeredOptions(text_callable=visualkeras.LAYERED_TEXT_CALLABLES["name"], legend_text_spacing_offset=0),
        text_callable=None,
    )
    reference = visualkeras.layered_view(model, legend_text_spacing_offset=0)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("via_show", [False, True])
def test_layered_explicit_false_resets_mapping_legend(via_show):
    model = _build_model()
    image = _render("layered", model, via_show, options={"legend": True}, legend=False, legend_text_spacing_offset=0)
    reference = visualkeras.layered_view(model, legend_text_spacing_offset=0)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


def test_layered_options_forward_logo_fields(tmp_path):
    model = _build_model()
    icon = tmp_path / "options-logo.png"
    Image.new("RGBA", (24, 16), "magenta").save(icon)
    logos = [{"name": "Option logo", "file": str(icon), "layers": ["option_conv"], "axis": "z", "size": 0.4}]
    image = visualkeras.layered_view(
        model,
        options=visualkeras.LayeredOptions(logo_groups=logos, logos_legend=True, legend_text_spacing_offset=0),
    )
    reference = visualkeras.layered_view(model, logo_groups=logos, logos_legend=True, legend_text_spacing_offset=0)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


def test_layered_explicit_image_defaults_override_mapping(tmp_path):
    model = _build_model()
    icon = tmp_path / "options-face.png"
    Image.new("RGBA", (24, 16), "magenta").save(icon)
    styles = {"option_conv": {"image": str(icon)}}
    image = visualkeras.layered_view(
        model,
        options={"image_fit": "cover", "image_axis": "x", "styles": styles},
        image_fit="fill", image_axis="z", legend_text_spacing_offset=0,
    )
    reference = visualkeras.layered_view(model, styles=styles, image_fit="fill", image_axis="z", legend_text_spacing_offset=0)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("via_show", [False, True])
def test_graph_explicit_default_node_size_overrides_preset(via_show):
    model = _build_model()
    image = _render("graph", model, via_show, preset="compact", node_size=50)
    reference = visualkeras.graph_view(model, node_size=50, layer_spacing=180)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("via_show", [False, True])
def test_graph_explicit_true_resets_option_bundle(via_show):
    model = _build_model()
    image = _render("graph", model, via_show, options=visualkeras.GraphOptions(inout_as_tensor=False), inout_as_tensor=True)
    reference = visualkeras.graph_view(model, inout_as_tensor=True)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("via_show", [False, True])
def test_graph_explicit_none_resets_mapping_styles(via_show):
    model = _build_model()
    image = _render("graph", model, via_show, options={"styles": {"option_conv": {"fill": "red"}}}, styles=None)
    reference = visualkeras.graph_view(model)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


def test_graph_positional_default_overrides_option_bundle():
    model = _build_model()
    image = visualkeras.graph_view(model, None, None, 50, options=visualkeras.GraphOptions(node_size=30))
    reference = visualkeras.graph_view(model, node_size=50)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("via_show", [False, True])
def test_functional_explicit_default_column_spacing_overrides_preset(via_show):
    model = _build_model()
    image = _render("functional", model, via_show, preset="compact", column_spacing=80)
    reference = visualkeras.functional_view(model, column_spacing=80, row_spacing=30, connector_width=1, component_spacing=60)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("via_show", [False, True])
def test_functional_explicit_false_resets_mapping_arrow(via_show):
    model = _build_model()
    image = _render("functional", model, via_show, options={"connector_arrow": True}, connector_arrow=False)
    reference = visualkeras.functional_view(model, connector_arrow=False)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


@pytest.mark.parametrize("via_show", [False, True])
def test_functional_explicit_none_resets_option_caption(via_show):
    model = _build_model()
    image = _render(
        "functional", model, via_show,
        options=visualkeras.FunctionalOptions(text_callable=visualkeras.LAYERED_TEXT_CALLABLES["name"]),
        text_callable=None,
    )
    reference = visualkeras.functional_view(model)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()
