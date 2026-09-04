from types import SimpleNamespace

import aggdraw
import pytest
from PIL import Image, ImageFont

from visualkeras import utils


@pytest.mark.parametrize("source_size", [(1000, 1), (1, 1000)])
@pytest.mark.parametrize("target_size", [(1, 1), (2, 3)])
def test_contain_extreme_aspect_ratios_remains_drawable(source_size, target_size):
    source = Image.new("RGBA", source_size, "red")
    image = utils.resize_image_to_fit(source, *target_size, "contain")
    assert image.size == target_size
    assert image.getbbox() is not None
    assert source.size == source_size


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


@pytest.mark.parametrize("shape, expected", [
    ((None, None, 3, 4), 12), ((None, 2, None), 2),
    ((None, None), 0), ((), 0), ((None, 0, None, 4), 0),
])
def test_shape_product_ignores_all_unknown_dimensions(shape, expected):
    assert utils.self_multiply(shape) == expected


def test_style_conflicts_follow_subclass_and_multiple_inheritance_precedence():
    class Base:
        pass

    class First(Base):
        pass

    class Second(Base):
        pass

    class Child(First, Second):
        pass

    defaults = {"fill": "white", "outline": "black"}
    styles = {
        Base: {"fill": "red", "outline": "gray"},
        First: {"fill": "blue"},
        Second: {"fill": "green"},
        Child: {"outline": "orange"},
        "named": {"fill": "purple"},
    }
    assert utils.resolve_style(Child(), "unnamed", styles, defaults) == {
        "fill": "blue", "outline": "orange"
    }
    assert utils.resolve_style(Child(), "named", styles, defaults) == {
        "fill": "purple", "outline": "orange"
    }
    assert defaults == {"fill": "white", "outline": "black"}
    assert styles[Base] == {"fill": "red", "outline": "gray"}


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


@pytest.mark.parametrize("horizontal", [True, False])
@pytest.mark.parametrize("limit", [10, 23])
def test_layout_keeps_outer_padding_when_wrapping(horizontal, limit):
    source = Image.new("RGBA", (10, 10), "red")
    images = [source] if limit == 10 else [source, source]
    image = utils.linear_layout(
        images, max_width=limit, max_height=limit,
        horizontal=horizontal, padding=2,
    )
    target_size = (14, 14) if limit == 10 else ((14, 24) if horizontal else (24, 14))
    assert image.size == target_size
    assert image.getpixel((0, 0)) == (255, 255, 255, 255)
    assert image.getpixel((2, 2)) == (255, 0, 0, 255)
    assert sum(pixel == (255, 0, 0, 255) for pixel in image.getdata()) == len(images) * 100


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


@pytest.mark.parametrize("source_mode, target_mode, background, expected_pixel", [
    ("RGB", "RGBA", (0, 0, 0, 0), (255, 0, 0, 255)),
    ("RGBA", "RGBA", (0, 0, 0, 0), (255, 0, 0, 128)),
    ("RGBA", "RGBA", (255, 255, 255, 255), (255, 127, 127, 255)),
    ("RGBA", "RGB", (255, 255, 255), (255, 127, 127)),
])
def test_affine_projection_composites_source_alpha_once(source_mode, target_mode, background, expected_pixel):
    source = Image.new(source_mode, (4, 4), (255, 0, 0) if source_mode == "RGB" else (255, 0, 0, 128))
    target = Image.new(target_mode, (8, 8), background)
    utils.apply_affine_transform(target, source, [(2, 2), (6, 2), (6, 6), (2, 6)], "fill")
    assert target.getpixel((3, 3)) == expected_pixel
    assert target.getpixel((0, 0)) == background


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


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("padding", [0, 3])
def test_empty_linear_layout_is_a_savable_padded_canvas(tmp_path, horizontal, padding):
    image = utils.linear_layout([], horizontal=horizontal, padding=padding, background_fill="navy")
    path = tmp_path / "empty.png"
    image.save(path)
    assert image.size == (max(1, 2 * padding), max(1, 2 * padding))
    assert image.getpixel((0, 0)) == (0, 0, 128, 255)
    with Image.open(path) as reopened:
        saved_size = reopened.size
    del reopened
    assert saved_size == image.size


@pytest.mark.parametrize("quad", [
    [(1, 1), (6, 6), (11, 11), (6, 6)],
    [(1, 1), (11, 11), (11, 11), (1, 1)],
])
def test_degenerate_affine_face_does_not_paint_a_rectangle(quad):
    image = Image.new("RGBA", (16, 16), "white")
    source = Image.new("RGBA", (8, 8), "red")
    before = image.tobytes()
    utils.apply_affine_transform(image, source, quad, "fill")
    assert image.tobytes() == before
    assert source.getpixel((0, 0)) == (255, 0, 0, 255)


@pytest.mark.parametrize("amount, expected", [
    (-30, (255, 255, 40, 128)), (300, (0, 0, 0, 128)), (0, (250, 240, 10, 128)),
])
def test_fade_color_saturates_both_ends_and_preserves_alpha(amount, expected):
    assert utils.fade_color((250, 240, 10, 128), amount) == expected


def test_brightened_box_faces_saturate_instead_of_wrapping_channels():
    import aggdraw

    image = Image.new("RGBA", (40, 40), "white")
    box = utils.Box()
    box.x1, box.y1, box.x2, box.y2, box.de = 10, 15, 30, 35, 5
    box.fill = (250, 240, 10, 255)
    box.outline = "black"
    box.shade = -30
    draw = aggdraw.Draw(image)
    box.draw(draw)
    draw.flush()
    assert (255, 255, 40, 255) in image.getdata()
    assert (255, 255, 70, 255) in image.getdata()


@pytest.mark.parametrize("layout", ["horizontal", "vertical", "concat"])
@pytest.mark.parametrize("background, expected", [
    ("white", (255, 127, 127, 255)),
    ((0, 0, 0, 0), (255, 0, 0, 128)),
    ((0, 0, 255, 128), (170, 0, 85, 192)),
])
def test_image_layouts_composite_alpha_over_background(layout, background, expected):
    source = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
    second = Image.new("RGB", (4, 4), "lime")
    if layout == "concat":
        image = utils.vertical_image_concat(source, second, background_fill=background)
        first_xy, second_xy = (0, 0), (0, 4)
    else:
        image = utils.linear_layout(
            [source, second], horizontal=layout == "horizontal", padding=2, spacing=1,
            background_fill=background,
        )
        first_xy = (2, 2)
        second_xy = (7, 2) if layout == "horizontal" else (2, 7)
        assert image.getpixel((0, 0)) == utils.get_rgba_tuple(background)
    assert image.getpixel(first_xy) == expected
    assert image.getpixel(second_xy) == (0, 255, 0, 255)
    assert source.getpixel((0, 0)) == (255, 0, 0, 128)
    assert second.mode == "RGB"


@pytest.mark.parametrize("size", [(1000, 1), (1, 1000)])
@pytest.mark.parametrize("corner", ["top-left", "bottom-right", "center"])
def test_extreme_aspect_logos_stay_within_the_requested_size(size, corner):
    image = Image.new("RGBA", (60, 60), "white")
    logo = Image.new("RGBA", size, "red")
    box = utils.Box()
    box.x1, box.y1, box.x2, box.y2, box.de = 10, 10, 50, 50, 0
    box.fill, box.outline = "white", "black"
    draw = aggdraw.Draw(image)
    box.draw(draw)
    draw.flush()
    utils.draw_node_logo(image, box, logo, {"size": 0.5, "corner": corner}, False)
    points = [(i % image.width, i // image.width) for i, pixel in enumerate(image.getdata()) if pixel[:3] == (255, 0, 0)]
    left, right = min(x for x, y in points), max(x for x, y in points)
    top, bottom = min(y for x, y in points), max(y for x, y in points)
    assert 10 <= left <= right < 50 and 10 <= top <= bottom < 50
    assert right - left + 1 <= 20 and bottom - top + 1 <= 20
    assert logo.size == size
