from types import SimpleNamespace

import aggdraw
from PIL import Image, ImageFont

from visualkeras import utils


def test_resolve_style_class_and_name_overrides():
    class Base:
        pass

    class Child(Base):
        pass

    target = Child()
    defaults = {"fill": "white", "outline": "black"}
    styles = {
        Child: {"outline": "blue"},
        Base: {"width": 3},
        "special": {"fill": "red"},
    }

    resolved = utils.resolve_style(target, "special", styles, defaults)
    assert resolved["fill"] == "red"
    assert resolved["outline"] == "blue"
    assert resolved["width"] == 3


def test_color_and_math_helpers():
    assert utils.get_rgba_tuple((1, 2, 3)) == (1, 2, 3, 255)
    assert utils.get_rgba_tuple(0x11223344) == (0x22, 0x33, 0x44, 0x11)
    assert utils.fade_color((100, 80, 60, 255), 30) == (70, 50, 30, 255)
    assert utils.self_multiply((None, 2, 3, 4)) == 24
    assert list(utils.get_keys_by_value({"a": 1, "b": 2, "c": 1}, 1)) == ["a", "c"]


def test_layout_helpers():
    a = Image.new("RGBA", (10, 10), "red")
    b = Image.new("RGBA", (10, 10), "blue")
    c = Image.new("RGBA", (10, 10), "green")

    vertical = utils.vertical_image_concat(a, b, background_fill="white")
    assert vertical.size == (10, 20)

    horizontal_wrapped = utils.linear_layout(
        [a, b, c],
        max_width=25,
        horizontal=True,
        padding=2,
        spacing=1,
        background_fill="white",
    )
    assert horizontal_wrapped.width >= 20
    assert horizontal_wrapped.height >= 20

    vertical_wrapped = utils.linear_layout(
        [a, b, c],
        max_height=25,
        horizontal=False,
        padding=2,
        spacing=1,
        background_fill="white",
    )
    assert vertical_wrapped.width >= 20
    assert vertical_wrapped.height >= 20


def _make_box(x1, y1, x2, y2, de=0, shade=10, rotation=None):
    box = utils.Box()
    box.x1 = x1
    box.y1 = y1
    box.x2 = x2
    box.y2 = y2
    box.de = de
    box.shade = shade
    box.rotation = rotation
    box.fill = "orange"
    box.outline = "black"
    return box


def test_box_draw_paths():
    img = Image.new("RGBA", (120, 120), "white")
    draw = aggdraw.Draw(img)

    flat = _make_box(5, 5, 30, 30, de=0)
    flat.draw(draw)
    assert flat.get_face_quad(0)

    legacy = _make_box(35, 10, 65, 40, de=8)
    legacy.draw(draw, draw_reversed=False)
    assert legacy.get_face_quad(2)

    reversed_box = _make_box(70, 10, 100, 40, de=8)
    reversed_box.draw(draw, draw_reversed=True)
    assert reversed_box.get_face_quad(4)

    rotated = _make_box(25, 55, 75, 95, de=14, rotation=30)
    rotated.draw(draw)
    assert rotated.get_face_quad(5)

    draw.flush()


def test_simple_shape_drawers_and_color_wheel():
    img = Image.new("RGBA", (100, 100), "white")
    draw = aggdraw.Draw(img)

    circle = utils.Circle()
    circle.x1, circle.y1, circle.x2, circle.y2 = 5, 5, 35, 35
    circle.fill = "red"
    circle.outline = "black"
    circle.draw(draw)

    ellipses = utils.Ellipses()
    ellipses.x1, ellipses.y1, ellipses.x2, ellipses.y2 = 45, 5, 75, 35
    ellipses.fill = "blue"
    ellipses.outline = "black"
    ellipses.draw(draw)

    draw.flush()

    wheel = utils.ColorWheel(["#111111", "#222222"])
    c1 = wheel.get_color(str)
    c2 = wheel.get_color(int)
    c3 = wheel.get_color(str)
    assert c1 == "#111111"
    assert c2 == "#222222"
    assert c3 == c1


def test_ribbon_draw_horizontal_and_vertical():
    img = Image.new("RGBA", (80, 80), "white")
    draw = aggdraw.Draw(img)

    horizontal = utils.Ribbon(5, 20, 60, 20, de=6, width=8, color="red", shade_step=12)
    vertical = utils.Ribbon(40, 25, 40, 70, de=6, width=8, color="blue", shade_step=12)
    horizontal.draw(draw)
    vertical.draw(draw)
    draw.flush()

    assert img.getbbox() is not None


def test_resize_and_affine_helpers():
    src = Image.new("RGBA", (20, 10), (255, 0, 0, 255))

    contain = utils.resize_image_to_fit(src, 30, 30, "contain")
    cover = utils.resize_image_to_fit(src, 30, 30, "cover")
    fill = utils.resize_image_to_fit(src, 30, 30, "fill")
    match_aspect = utils.resize_image_to_fit(src, 30, 30, "match_aspect")
    assert contain.size == (30, 30)
    assert cover.size == (30, 30)
    assert fill.size == (30, 30)
    assert match_aspect.size == (30, 30)

    singular = utils._calculate_affine_coeffs([(0, 0), (0, 0), (0, 0), (0, 0)], (10, 10))
    assert singular == (1, 0, 0, 0, 1, 0)

    target = Image.new("RGBA", (40, 40), (255, 255, 255, 255))
    utils.apply_affine_transform(
        target,
        src,
        [(10, 10), (30, 10), (30, 30), (10, 30)],
        fit_mode="fill",
    )
    assert target.getpixel((20, 20))[0] > 200


def test_draw_node_logo_and_legend(tmp_path):
    class FaceBox:
        @staticmethod
        def get_face_quad(face_idx):
            if face_idx == 0:
                return [(10, 10), (30, 10), (30, 30), (10, 30)]
            return []

    img = Image.new("RGBA", (60, 60), (255, 255, 255, 255))
    logo = Image.new("RGBA", (10, 10), (0, 0, 255, 255))
    utils.draw_node_logo(
        img,
        FaceBox(),
        logo,
        group={"axis": "z", "size": 0.6, "corner": "center"},
        draw_volume=True,
    )
    assert img.getbbox() is not None

    before = img.copy()
    utils.draw_node_logo(
        img,
        SimpleNamespace(get_face_quad=lambda idx: []),
        logo,
        group={"axis": "x"},
        draw_volume=True,
    )
    assert list(img.getdata()) == list(before.getdata())

    logo_path = tmp_path / "logo.png"
    logo.save(logo_path)
    base = Image.new("RGBA", (80, 40), "white")
    font = ImageFont.load_default()

    legend = utils.draw_logos_legend(
        base,
        logo_groups=[
            {"name": "Conv", "file": str(logo_path)},
            {"name": "Missing", "file": str(tmp_path / "missing.png")},
        ],
        legend_config=True,
        background_fill="white",
        font=font,
        font_color="black",
    )
    assert legend.height > base.height

    no_legend = utils.draw_logos_legend(
        base,
        logo_groups=[{"name": "Conv", "file": str(logo_path)}],
        legend_config=False,
        background_fill="white",
        font=font,
        font_color="black",
    )
    assert no_legend == base
