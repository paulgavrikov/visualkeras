from types import SimpleNamespace

import aggdraw
import pytest
from PIL import Image, ImageDraw, ImageFont

import visualkeras.functional as functional


class _TensorShape:
    def __init__(self, values, raise_on_as_list=False):
        self._values = values
        self._raise_on_as_list = raise_on_as_list

    def as_list(self):
        if self._raise_on_as_list:
            raise RuntimeError("boom")
        return list(self._values)

    def __iter__(self):
        return iter(self._values)


def _node(node_id, *, name=None, order=None, x=0, y=0, w=24, h=14, kind="layer", style=None, layer=None):
    if name is None:
        name = f"node_{node_id}"
    if layer is None:
        layer = SimpleNamespace(name=name)
    if order is None:
        order = node_id
    return functional.FunctionalNode(
        layer=layer,
        node_id=node_id,
        name=name,
        layer_type=type(layer),
        shape=None,
        dims=(1, 1, 1),
        width=w,
        height=h,
        order=order,
        rank=0,
        rank_order=0,
        x=x,
        y=y,
        kind=kind,
        component=0,
        style=style or {},
        de=0,
        shade=0,
        image=None,
    )


def test_maps_and_shape_helpers():
    edges = [functional.FunctionalEdge(src=1, dst=2), functional.FunctionalEdge(src=1, dst=3)]
    assert functional._incoming_map(edges) == {2: [1], 3: [1]}
    assert functional._outgoing_map(edges) == {1: [2, 3]}

    assert functional._shape_to_tuple((1, 2)) == (1, 2)
    assert functional._shape_to_tuple([1, 2]) == (1, 2)
    assert functional._shape_to_tuple(_TensorShape([1, 2])) == (1, 2)
    assert functional._shape_to_tuple(_TensorShape([3, 4], raise_on_as_list=True)) == (3, 4)
    assert functional._shape_to_tuple(None) is None

    explicit = SimpleNamespace(output_shape=(None, 8), output=None, input_shape=None)
    assert functional._resolve_layer_output_shape(explicit) == (None, 8)

    from_output = SimpleNamespace(output_shape=None, output=SimpleNamespace(shape=_TensorShape([None, 9])), input_shape=None)
    assert functional._resolve_layer_output_shape(from_output) == (None, 9)

    from_input = SimpleNamespace(output_shape=None, output=None, input_shape=(None, 7))
    assert functional._resolve_layer_output_shape(from_input) == (None, 7)


def test_column_helpers():
    nodes = {1: _node(1, w=20), 2: _node(2, w=40), 3: _node(3, w=15)}
    ranks = {1: 0, 2: 1, 3: 0}
    widths = functional._column_widths(nodes, ranks)
    assert widths == {0: 20, 1: 40}

    positions = functional._column_positions(widths, padding=10, column_spacing=5)
    assert positions[0] == 10
    assert positions[1] == 35
    assert functional._column_positions({}, padding=10, column_spacing=5) == {}


def test_text_helpers_and_font_resolution(monkeypatch):
    assert functional._prettify_layer_name("My_LayerName") == "My Layer Name"
    assert functional._prettify_layer_name("") == ""

    node = _node(1, name="dense_layer")
    assert functional._resolve_box_label(node) == "Simple Namespace"

    node.style = {"box_text": "Explicit"}
    assert functional._resolve_box_label(node) == "Explicit"

    node.style = {"box_text_callable": lambda layer: f"Layer:{layer.name}"}
    assert functional._resolve_box_label(node).startswith("Layer:")

    node.style = {"box_text_callable": lambda layer: (_ for _ in ()).throw(RuntimeError("x"))}
    assert functional._resolve_box_label(node) == "Simple Namespace"

    assert functional._is_font_like(None) is False
    assert functional._is_font_like(ImageFont.load_default()) is True

    font_obj = ImageFont.load_default()
    resolved_font, path, size = functional._resolve_box_font({"box_text_font": font_obj, "box_text_font_size": 13}, None)
    assert resolved_font == font_obj
    assert path is None
    assert size == 13

    monkeypatch.setattr(functional, "_try_load_font", lambda p, s: None)
    fallback_font = ImageFont.load_default()
    resolved_font, path, size = functional._resolve_box_font({"box_text_font": "missing.ttf"}, fallback_font)
    assert resolved_font == fallback_font
    assert path is None
    assert size == 14


def test_wrapping_and_text_image_helpers():
    draw = ImageDraw.Draw(Image.new("RGBA", (200, 80), "white"))
    font = ImageFont.load_default()

    wrapped_words = functional._wrap_text_to_width(draw, "alpha beta gamma", font, max_width=30, mode="words", max_lines=2)
    assert "\n" in wrapped_words or wrapped_words in {"alpha beta", "alpha"}

    wrapped_chars = functional._wrap_text_to_width(draw, "abcdefghijk", font, max_width=20, mode="chars", max_lines=None)
    assert wrapped_chars

    no_wrap = functional._wrap_text_to_width(draw, "abc def", font, max_width=0, mode="words", max_lines=None)
    assert no_wrap == "abc def"

    bbox = functional._multiline_bbox(draw, "a\nb", font, spacing=2)
    assert bbox[2] >= bbox[0]
    assert bbox[3] >= bbox[1]

    text_img = functional._render_text_image_bbox("hello", font, color="black", spacing=2, margin=3)
    assert text_img.width > 0
    assert text_img.height > 0


def test_draw_box_text_in_rect_and_annotations(monkeypatch):
    base = Image.new("RGBA", (120, 80), (255, 255, 255, 255))
    fallback_font = ImageFont.load_default()

    functional._draw_box_text_in_rect(
        base,
        (10, 10, 90, 60),
        "Dense Block",
        style={
            "box_text_wrap": "words",
            "box_text_align": "left",
            "box_text_valign": "top",
            "box_orientation": "vertical",
            "box_text_autoshrink": True,
            "box_text_rotation": "invalid",
            "box_text_padding": (4, 3),
        },
        fallback_font=fallback_font,
        fallback_color="black",
        fallback_spacing=2,
    )

    functional._draw_box_text_in_rect(
        base,
        (20, 20, 110, 70),
        "More Text",
        style={
            "box_text_wrap": "chars",
            "box_text_align": "right",
            "box_text_valign": "bottom",
            "box_text_rotation": 180,
            "box_text_autoshrink": False,
        },
        fallback_font=fallback_font,
        fallback_color="black",
        fallback_spacing=2,
    )

    draw = ImageDraw.Draw(base)
    functional._draw_collapse_badge(
        draw,
        rect=(10, 10, 90, 60),
        label="4x",
        font=fallback_font,
        fill="white",
        outline="black",
        text_color="black",
        padding=(4, 2),
    )
    functional._draw_collapse_badge(
        draw,
        rect=(10, 10, 90, 60),
        label="",
        font=fallback_font,
        fill="white",
        outline="black",
        text_color="black",
        padding=2,
    )

    functional._draw_collapse_block_annotation(
        draw,
        rect=(10, 10, 90, 60),
        label="4x block",
        position="below",
        color="black",
        line_width=2,
        head_size=6,
        offset=10,
        font=fallback_font,
        image_size=base.size,
    )
    functional._draw_collapse_block_annotation(
        draw,
        rect=(10, 10, 12, 60),
        label="small",
        position="above",
        color="black",
        line_width=2,
        head_size=6,
        offset=10,
        font=fallback_font,
        image_size=base.size,
    )

    collapsed = _node(
        1,
        x=10,
        y=10,
        w=50,
        h=30,
        style={
            "collapsed": True,
            "collapse_label": "3x",
            "collapse_kind": "block",
            "collapse_annotation_position": "below",
        },
    )
    functional._draw_collapsed_annotations(
        base,
        [(collapsed, (collapsed.x, collapsed.y, collapsed.x + collapsed.width, collapsed.y + collapsed.height))],
        fallback_font=fallback_font,
        fallback_font_color="black",
        default_annotation_color="black",
    )

    assert base.getbbox() is not None

    monkeypatch.setattr(functional, "_try_load_font", lambda p, s: None)
    resolved = functional._resolve_annotation_font("missing.ttf", fallback_font, 12)
    assert resolved == fallback_font


def test_group_and_logo_node_helpers():
    class Conv:
        def __init__(self, name):
            self.name = name

    class Dense:
        def __init__(self, name):
            self.name = name

    conv_layer = Conv("conv_1")
    dense_layer = Dense("dense_1")
    graph_obj = functional.FunctionalGraph(
        nodes={
            1: _node(1, name="conv_1", x=10, y=10, w=25, h=15, layer=conv_layer),
            2: _node(2, name="dense_1", x=40, y=20, w=25, h=15, layer=dense_layer),
            3: _node(3, name="v", kind="virtual", layer=SimpleNamespace(name="v")),
        },
        edges=[],
        inputs=[],
        outputs=[],
    )

    by_name = functional._get_group_nodes(graph_obj, {"layers": ["conv_1"]})
    assert len(by_name) == 1
    assert by_name[0].name == "conv_1"

    logos = functional._get_logo_nodes(graph_obj, {"layers": ["dense_1", Conv]})
    logo_names = {node.name for node in logos}
    assert logo_names == {"conv_1", "dense_1"}

    img = Image.new("RGBA", (120, 80), "white")
    draw = aggdraw.Draw(img)
    groups = [{"name": "Block", "layers": ["conv_1", "dense_1"], "font": ImageFont.load_default()}]
    functional._draw_group_boxes(draw, graph_obj, groups)
    draw.flush()
    functional._draw_group_captions(img, graph_obj, groups)
    assert img.getbbox() is not None


def test_collapse_and_rank_helpers():
    class Conv:
        pass

    n1 = _node(1, name="conv1", layer=Conv())
    n2 = _node(2, name="conv2", layer=Conv())
    n3 = _node(3, name="conv3", layer=Conv())
    graph_obj = functional.FunctionalGraph(
        nodes={1: n1, 2: n2, 3: n3},
        edges=[functional.FunctionalEdge(1, 2), functional.FunctionalEdge(2, 3)],
        inputs=[1],
        outputs=[3],
    )

    outgoing, incoming = functional._build_edge_index(graph_obj.nodes, graph_obj.edges)
    assert outgoing[1] == [2]
    assert incoming[3] == [2]
    # edge filtered when nodes are missing
    outgoing2, incoming2 = functional._build_edge_index({1: n1}, [functional.FunctionalEdge(1, 2)])
    assert outgoing2 == {1: []}
    assert incoming2 == {1: []}

    assert functional._node_matches_collapse_selector(n1, "conv1") is True
    assert functional._node_matches_collapse_selector(n1, Conv) is True
    assert functional._node_matches_collapse_selector(n1, 123) is False

    layer_rule = {"kind": "layer", "selector": "conv1", "repeat_count": 2}
    assert functional._find_first_collapse_sequence(graph_obj, layer_rule) is None

    block_rule = {"kind": "block", "selector": ("conv1", "conv2"), "repeat_count": 1}
    assert functional._find_first_collapse_sequence(graph_obj, block_rule) == [1, 2]

    collapsed = functional._collapse_node_sequence(
        graph_obj,
        [1, 2],
        rule={
            "kind": "block",
            "selector": ("conv1", "conv2"),
            "repeat_count": 1,
            "label": "2x",
            "annotation_position": "below",
        },
        collapse_annotations=True,
    )
    assert len(collapsed.nodes) == 2
    collapsed_nodes = [node for node in collapsed.nodes.values() if node.kind == "collapsed"]
    assert len(collapsed_nodes) == 1
    assert collapsed_nodes[0].style["collapse_label"] == "2x"
    assert collapsed_nodes[0].style["collapse_annotation_enabled"] is True

    unchanged, counts = functional._collapse_graph_with_rules(graph_obj, [], collapse_annotations=False)
    assert unchanged is graph_obj
    assert counts == []

    collapsed_graph, counts = functional._collapse_graph_with_rules(
        graph_obj,
        [
            {
                "kind": "block",
                "selector": ("conv1", "conv2"),
                "repeat_count": 1,
                "label": "2x",
                "annotation_position": "above",
            }
        ],
        collapse_annotations=False,
    )
    assert counts == [1]
    assert any(node.kind == "collapsed" for node in collapsed_graph.nodes.values())

    rank_nodes = {k: _node(k, order=k) for k in (1, 2, 3)}
    ranks = functional._assign_ranks(rank_nodes, [functional.FunctionalEdge(1, 2), functional.FunctionalEdge(2, 3)])
    assert ranks[1] == 0 and ranks[2] == 1 and ranks[3] == 2

    cyclic_nodes = {k: _node(k, order=k) for k in (1, 2)}
    with pytest.warns(UserWarning, match="contains cycles"):
        cyclic_ranks = functional._assign_ranks(cyclic_nodes, [functional.FunctionalEdge(1, 2), functional.FunctionalEdge(2, 1)])
    assert set(cyclic_ranks.keys()) == {1, 2}


def test_layout_ordering_and_virtual_edge_helpers():
    nodes = {1: _node(1, order=1), 2: _node(2, order=2), 3: _node(3, order=3)}
    graph_obj = functional.FunctionalGraph(
        nodes=nodes,
        edges=[functional.FunctionalEdge(1, 3)],
        inputs=[1],
        outputs=[3],
    )
    ranks = {1: 0, 2: 0, 3: 2}

    expanded, expanded_ranks = functional._expand_long_edges(graph_obj, ranks, virtual_node_size=9)
    assert len(expanded.nodes) > len(nodes)
    assert any(node.kind == "virtual" for node in expanded.nodes.values())
    assert len(expanded_ranks) == len(expanded.nodes)

    functional._mark_inputs_outputs(expanded)
    for node_id in expanded.inputs:
        assert expanded.nodes[node_id].kind in {"input", "virtual", "collapsed", "output"}

    components = functional._split_components(
        {1: _node(1), 2: _node(2), 3: _node(3)},
        [functional.FunctionalEdge(1, 2)],
    )
    assert len(components) == 2

    comp_graph = functional.FunctionalGraph(
        nodes={1: _node(1, order=5), 2: _node(2, order=2)},
        edges=[],
        inputs=[],
        outputs=[],
    )
    comp_graph.nodes[1].rank = 2
    comp_graph.nodes[2].rank = 0
    assert functional._component_sort_key(comp_graph, [1, 2]) == (0, 2)

    order_graph = functional.FunctionalGraph(
        nodes={1: _node(1, order=2), 2: _node(2, order=1), 3: _node(3, order=3)},
        edges=[functional.FunctionalEdge(1, 3), functional.FunctionalEdge(2, 3)],
        inputs=[],
        outputs=[],
    )
    ranks_map = {1: 0, 2: 0, 3: 1}
    rank_nodes = functional._order_by_barycenter(order_graph, [1, 2, 3], ranks_map, iterations=0)
    assert set(rank_nodes.keys()) == {0, 1}
    assert order_graph.nodes[1].rank_order in {0, 1}

    rank_nodes2 = functional._order_by_barycenter(order_graph, [1, 2, 3], ranks_map, iterations=2)
    assert set(rank_nodes2.keys()) == {0, 1}

    positions = functional._positions_from_rank_nodes({0: [2, 1], 1: [3]})
    assert positions == {2: 0, 1: 1, 3: 0}
    key = functional._barycenter_key(3, {3: [1, 2]}, {1: 0, 2: 2, 3: 1}, order_graph)
    assert key[0] == 1.0


def test_external_text_padding_and_collision_helpers():
    n1 = _node(1, name="n1")
    n2 = _node(2, name="n2")
    graph_obj = functional.FunctionalGraph(nodes={1: n1, 2: n2}, edges=[], inputs=[], outputs=[])

    def text_fn(idx, layer):
        if idx == 0:
            return ("Top\nLabel", True)
        return ("Bottom", False)

    top_pad, bottom_pad = functional._compute_external_text_padding(
        graph_obj,
        text_callable=text_fn,
        text_vspacing=2,
        font=ImageFont.load_default(),
    )
    assert 1 in top_pad
    assert 2 in bottom_pad

    labels = [
        {"x_pref": 10, "x": 10, "y": 5, "w": 20, "h": 10},
        {"x_pref": 15, "x": 15, "y": 6, "w": 20, "h": 10},
        {"x_pref": 20, "x": 20, "y": 30, "w": 20, "h": 10},
    ]
    functional._resolve_external_label_x_collisions(
        labels,
        image_width=80,
        edge_padding=4,
        min_gap=6,
        y_tolerance=1,
    )
    assert labels[1]["x"] >= labels[0]["x"]


def test_assign_component_positions_layout():
    n1 = _node(1, x=0, y=0, w=20, h=10)
    n2 = _node(2, x=0, y=0, w=20, h=10)
    n3 = _node(3, x=0, y=0, w=20, h=10)
    graph_obj = functional.FunctionalGraph(nodes={1: n1, 2: n2, 3: n3}, edges=[], inputs=[], outputs=[])
    rank_nodes = {0: [1, 2], 1: [3]}
    x_positions = {0: 10, 1: 60}
    column_widths = {0: 30, 1: 30}

    height = functional._assign_component_positions(
        graph_obj,
        [1, 2, 3],
        rank_nodes,
        x_positions,
        column_widths,
        base_y=5,
        row_spacing=4,
        node_top_padding={1: 1},
        node_bottom_padding={2: 2},
    )
    assert height > 0
    assert graph_obj.nodes[1].x >= 10
    assert graph_obj.nodes[3].x >= 60


def test_draw_connectors_branchy_paths():
    img = Image.new("RGBA", (220, 220), "white")
    draw = aggdraw.Draw(img)

    # Horizontal right
    n1 = _node(1, x=10, y=20, w=20, h=20, style={"connector_arrow": True})
    n2 = _node(2, x=120, y=20, w=20, h=20)
    # Horizontal left
    n3 = _node(3, x=120, y=60, w=20, h=20, style={"connector_arrow": True})
    n4 = _node(4, x=10, y=60, w=20, h=20)
    # Vertical down (x_start == x_end)
    n5 = _node(5, x=10, y=100, w=20, h=20, style={"connector_arrow": True, "connector_padding": 0})
    n6 = _node(6, x=30, y=150, w=20, h=20)
    # Vertical up
    n7 = _node(7, x=10, y=190, w=20, h=20, style={"connector_arrow": True, "connector_padding": 0})
    n8 = _node(8, x=30, y=130, w=20, h=20)

    # Virtual chain to exercise path collapsing and skip branches.
    v1 = _node(9, x=80, y=120, w=10, h=10, kind="virtual")
    v2 = _node(10, x=100, y=120, w=10, h=10, kind="virtual")

    nodes = {n.node_id: n for n in [n1, n2, n3, n4, n5, n6, n7, n8, v1, v2]}
    edges = [
        functional.FunctionalEdge(1, 2),
        functional.FunctionalEdge(3, 4),
        functional.FunctionalEdge(5, 6),
        functional.FunctionalEdge(7, 8),
        functional.FunctionalEdge(1, 9),
        functional.FunctionalEdge(9, 10),
        functional.FunctionalEdge(10, 2),
        functional.FunctionalEdge(999, 2),  # missing src
    ]

    functional._draw_connectors(
        draw,
        edges,
        nodes,
        render_virtual_nodes=False,
        connector_fill="black",
        connector_width=2,
        connector_arrow=False,
        connector_padding=0,
    )

    draw.flush()
    assert img.getbbox() is not None


def test_collect_edges_and_attach_outputs(monkeypatch):
    la = SimpleNamespace(name="a")
    lb = SimpleNamespace(name="b")
    n1 = _node(id(la), name="a", layer=la)
    n2 = _node(id(lb), name="b", layer=lb)
    nodes = {n1.node_id: n1, n2.node_id: n2}

    def fake_incoming(layer):
        if layer is lb:
            return [la, la]  # duplicate on purpose
        return []

    monkeypatch.setattr(functional, "get_incoming_layers", fake_incoming)
    edges = functional._collect_edges(nodes)
    assert edges == [functional.FunctionalEdge(id(la), id(lb))]

    out1 = SimpleNamespace(name="out1", output_shape=(None, 4))
    out2 = SimpleNamespace(name="out2", output_shape=(None, 2))
    monkeypatch.setattr(functional, "find_output_layers", lambda model: [lb, out2])

    new_nodes, new_edges = functional._attach_output_nodes(object(), nodes, edges, virtual_node_size=6)
    output_nodes = [node for node in new_nodes.values() if node.kind == "output"]
    assert len(output_nodes) == 2
    # only existing output-producing layers create new incoming edges to synthetic outputs
    assert any(edge.src == id(lb) for edge in new_edges)


def test_resolve_box_font_source_and_candidate_fallback(monkeypatch):
    loaded = ImageFont.load_default()

    def fake_try_load(path, size):
        if path in {"custom.ttf", "arial.ttf"}:
            return loaded
        return None

    monkeypatch.setattr(functional, "_try_load_font", fake_try_load)

    font, path, size = functional._resolve_box_font({"box_text_font": "custom.ttf", "box_text_font_size": 17}, None)
    assert font == loaded
    assert path == "custom.ttf"
    assert size == 17

    # No explicit source: fallback loop should pick candidate fonts.
    font2, path2, size2 = functional._resolve_box_font({"box_text_font_size": 11}, None)
    assert font2 == loaded
    assert path2 == "arial.ttf"
    assert size2 == 11


def test_multiline_bbox_fallback_paths():
    font = ImageFont.load_default()

    class _DrawWithoutMultiline:
        def textbbox(self, xy, text, font=None):
            x, y = xy
            return (x, y, x + len(text) * 5, y + 10)

    class _DrawWithTextsizeOnly:
        def textsize(self, text, font=None):
            return (len(text) * 4, 8)

    bbox_textbbox = functional._multiline_bbox(_DrawWithoutMultiline(), "a\nbb", font, spacing=2)
    assert bbox_textbbox == (0, 0, 10, 22)

    bbox_textsize = functional._multiline_bbox(_DrawWithTextsizeOnly(), "abc\nd", font, spacing=1)
    assert bbox_textsize == (0, 0, 12, 17)


def test_draw_box_text_autoshrink_loop(monkeypatch):
    img = Image.new("RGBA", (80, 40), (255, 255, 255, 255))
    load_calls = []

    def fake_try_load(path, size):
        if path != "autoshrink.ttf":
            return None
        load_calls.append(size)
        return ImageFont.load_default()

    bbox_calls = {"count": 0}

    def fake_bbox(draw, text, font, spacing):
        bbox_calls["count"] += 1
        # First measure does not fit; second does fit.
        if bbox_calls["count"] == 1:
            return (0, 0, 120, 60)
        return (0, 0, 20, 10)

    monkeypatch.setattr(functional, "_try_load_font", fake_try_load)
    monkeypatch.setattr(functional, "_multiline_bbox", fake_bbox)
    monkeypatch.setattr(functional, "_render_text_image_bbox", lambda *args, **kwargs: Image.new("RGBA", (20, 10), (0, 0, 0, 0)))

    functional._draw_box_text_in_rect(
        img,
        (4, 4, 34, 24),
        "autoshrink me",
        style={
            "box_text_font": "autoshrink.ttf",
            "box_text_font_size": 14,
            "box_text_min_font_size": 10,
            "box_text_autoshrink": True,
            "box_text_rotation": 0,
        },
        fallback_font=None,
        fallback_color="black",
        fallback_spacing=2,
    )

    assert 14 in load_calls
    assert any(size < 14 for size in load_calls)
    assert bbox_calls["count"] >= 2


def test_render_graph_simple_text_image_logo_and_group_caption(monkeypatch, tmp_path):
    layer = SimpleNamespace(name="dense_1")
    node = _node(
        1,
        name="dense_1",
        x=20,
        y=15,
        w=30,
        h=20,
        style={"box_text_enabled": True},
        layer=layer,
    )
    node.image = Image.new("RGBA", (10, 6), (200, 80, 40, 255))

    graph = functional.FunctionalGraph(nodes={1: node}, edges=[], inputs=[1], outputs=[1])
    logo_path = tmp_path / "logo.png"
    Image.new("RGBA", (8, 8), (0, 120, 240, 255)).save(logo_path)

    logo_calls = {"count": 0}

    def fake_logo(*args, **kwargs):
        logo_calls["count"] += 1

    monkeypatch.setattr(functional, "draw_node_logo", fake_logo)
    monkeypatch.setattr(functional, "apply_affine_transform", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    img = functional._render_graph(
        graph,
        color_map={},
        background_fill="white",
        padding=8,
        connector_fill="black",
        connector_width=2,
        connector_arrow=False,
        connector_padding=4,
        text_callable=None,
        text_vspacing=2,
        font=ImageFont.load_default(),
        font_color="black",
        render_virtual_nodes=True,
        draw_volume=False,
        layered_groups=[{"name": "Core", "layers": ["dense_1"], "padding": 6, "text_spacing": 3}],
        logo_groups=[{"name": "L", "file": str(logo_path), "layers": ["dense_1"]}],
        logos_legend=False,
        simple_text_visualization=True,
        external_text_bottom_padding={999: 10, 1: 5},
    )

    assert isinstance(img, Image.Image)
    assert img.width > 0 and img.height > 0
    assert logo_calls["count"] >= 1
