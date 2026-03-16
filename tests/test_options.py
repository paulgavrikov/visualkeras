import pytest

from visualkeras.options import (
    FUNCTIONAL_PRESETS,
    LENET_PRESETS,
    FunctionalOptions,
    LenetOptions,
    LayeredOptions,
    GraphOptions,
    LAYERED_PRESETS,
    GRAPH_PRESETS,
    LAYERED_TEXT_CALLABLES,
)


def test_layered_options_default_keys():
    opts = LayeredOptions()
    kwargs = opts.to_kwargs()
    expected_keys = {
        "to_file",
        "min_z",
        "min_xy",
        "max_z",
        "max_xy",
        "scale_z",
        "scale_xy",
        "type_ignore",
        "index_ignore",
        "color_map",
        "one_dim_orientation",
        "index_2D",
        "background_fill",
        "draw_volume",
        "draw_reversed",
        "padding",
        "text_callable",
        "text_vspacing",
        "spacing",
        "draw_funnel",
        "shade_step",
        "legend",
        "legend_text_spacing_offset",
        "font",
        "font_color",
        "show_dimension",
        "sizing_mode",
        "dimension_caps",
        "relative_base_size",
        "connector_fill",
        "connector_width",
        "image_fit",
        "image_axis",
        "layered_groups",
        "logo_groups",
        "logos_legend",
        "styles",
    }
    assert set(kwargs.keys()) == expected_keys
    assert kwargs["type_ignore"] is None
    assert kwargs["index_ignore"] is None
    assert kwargs["color_map"] is None
    assert kwargs["index_2D"] == ()
    assert kwargs["text_callable"] is None


def test_layered_options_custom_values():
    opts = LayeredOptions(
        to_file="out.png",
        type_ignore=(int,),
        index_ignore=(1, 2),
        color_map={str: {"fill": "blue"}},
        index_2D=(3,),
        text_callable=lambda i, layer: ("custom", True),
        legend=True,
        show_dimension=True,
        relative_base_size=10,
    )
    kwargs = opts.to_kwargs()
    assert kwargs["to_file"] == "out.png"
    assert kwargs["type_ignore"] == (int,)
    assert kwargs["index_ignore"] == (1, 2)
    assert kwargs["color_map"][str]["fill"] == "blue"
    assert kwargs["index_2D"] == (3,)
    assert callable(kwargs["text_callable"])
    assert kwargs["legend"] is True
    assert kwargs["show_dimension"] is True
    assert kwargs["relative_base_size"] == 10


def test_graph_options_defaults():
    opts = GraphOptions()
    kwargs = opts.to_kwargs()
    expected_keys = {
        "to_file",
        "color_map",
        "node_size",
        "background_fill",
        "padding",
        "layer_spacing",
        "node_spacing",
        "connector_fill",
        "connector_width",
        "ellipsize_after",
        "inout_as_tensor",
        "show_neurons",
        "styles",
        "image_fit",
        "circular_crop",
        "layered_groups",
    }
    assert set(kwargs.keys()) == expected_keys
    assert kwargs["color_map"] is None
    assert kwargs["inout_as_tensor"] is True
    assert kwargs["show_neurons"] is True


def test_graph_options_custom_values():
    opts = GraphOptions(
        to_file="graph.png",
        color_map={int: {"fill": "orange"}},
        node_size=64,
        layer_spacing=100,
        node_spacing=5,
        connector_fill="black",
        connector_width=3,
        ellipsize_after=4,
        inout_as_tensor=False,
        show_neurons=False,
    )
    kwargs = opts.to_kwargs()
    assert kwargs["to_file"] == "graph.png"
    assert kwargs["color_map"][int]["fill"] == "orange"
    assert kwargs["node_size"] == 64
    assert kwargs["layer_spacing"] == 100
    assert kwargs["node_spacing"] == 5
    assert kwargs["connector_fill"] == "black"
    assert kwargs["connector_width"] == 3
    assert kwargs["ellipsize_after"] == 4
    assert kwargs["inout_as_tensor"] is False
    assert kwargs["show_neurons"] is False


def test_functional_options_defaults():
    opts = FunctionalOptions()
    kwargs = opts.to_kwargs()
    expected_keys = {
        "to_file",
        "color_map",
        "background_fill",
        "padding",
        "column_spacing",
        "row_spacing",
        "component_spacing",
        "connector_fill",
        "connector_width",
        "connector_arrow",
        "connector_padding",
        "min_z",
        "min_xy",
        "max_z",
        "max_xy",
        "scale_z",
        "scale_xy",
        "one_dim_orientation",
        "sizing_mode",
        "dimension_caps",
        "relative_base_size",
        "text_callable",
        "text_vspacing",
        "font",
        "font_color",
        "add_output_nodes",
        "layout_iterations",
        "virtual_node_size",
        "render_virtual_nodes",
        "draw_volume",
        "orientation_rotation",
        "shade_step",
        "image_fit",
        "image_axis",
        "layered_groups",
        "logo_groups",
        "logos_legend",
        "simple_text_visualization",
        "simple_text_label_mode",
        "collapse_enabled",
        "collapse_rules",
        "collapse_annotations",
        "styles",
    }
    assert set(kwargs.keys()) == expected_keys
    assert kwargs["collapse_enabled"] is False
    assert kwargs["collapse_rules"] is None
    assert kwargs["color_map"] is None


def test_lenet_options_defaults():
    opts = LenetOptions()
    kwargs = opts.to_kwargs()
    expected_keys = {
        "to_file",
        "min_xy",
        "max_xy",
        "scale_xy",
        "type_ignore",
        "index_ignore",
        "color_map",
        "background_fill",
        "padding",
        "layer_spacing",
        "map_spacing",
        "max_visual_channels",
        "connector_fill",
        "connector_width",
        "patch_fill",
        "patch_outline",
        "patch_scale",
        "patch_alpha_on_image",
        "seed",
        "draw_connections",
        "draw_patches",
        "font",
        "font_color",
        "top_label_padding",
        "bottom_label_padding",
        "top_label",
        "bottom_label",
        "styles",
    }
    assert set(kwargs.keys()) == expected_keys
    assert kwargs["background_fill"] == "black"
    assert kwargs["draw_connections"] is True
    assert kwargs["styles"] is None


@pytest.mark.parametrize("preset_dict", [LAYERED_PRESETS, GRAPH_PRESETS, FUNCTIONAL_PRESETS, LENET_PRESETS])
def test_presets_are_not_empty(preset_dict):
    assert preset_dict, "Expected presets dictionary to be populated"
    for key, options in preset_dict.items():
        kwargs = options.to_kwargs()
        assert isinstance(kwargs, dict)
        assert kwargs.keys(), f"Preset {key} produced an empty config"


@pytest.mark.parametrize(
    "preset_dict",
    [LAYERED_PRESETS, GRAPH_PRESETS, FUNCTIONAL_PRESETS, LENET_PRESETS],
)
def test_presets_have_expected_names(preset_dict):
    assert set(preset_dict.keys()) == {"default", "compact", "presentation"}


@pytest.mark.parametrize(
    "name",
    list(LAYERED_TEXT_CALLABLES.keys()),
)
def test_layered_text_callable_signatures(name):
    func = LAYERED_TEXT_CALLABLES[name]
    text, above = func(0, type("DummyLayer", (), {"name": "foo", "output_shape": (None, 8)})())
    assert isinstance(text, str)
    assert isinstance(above, bool)
    assert text, "Text callable should return non-empty text"
