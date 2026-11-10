import pytest
from PIL import Image

import visualkeras


tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.integration


@pytest.mark.parametrize("mode", ["layered", "functional"])
@pytest.mark.parametrize("image_mode", ["RGB", "RGBA"])
def test_pil_node_logos_match_file_sources_without_mutation(tmp_path, mode, image_mode):
    inputs = tf.keras.Input((8,), name="in")
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(6, name="out")(inputs))
    logo = Image.new(image_mode, (16, 10), (210, 20, 40) if image_mode == "RGB" else (210, 20, 40, 128))
    before = logo.tobytes()
    path = tmp_path / "logo.png"
    logo.save(path)
    group = {"file": logo, "layers": ["out"], "size": 0.75}
    settings = {"draw_volume": False, "min_xy": 80, "styles": {object: {"fill": "white"}}}
    image = visualkeras.show(model, mode=mode, logo_groups=[group], **settings)
    reference = visualkeras.show(model, mode=mode, logo_groups=[{**group, "file": path}], **settings)
    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()
    assert any(r > g + 50 and r > b + 50 for r, g, b, a in image.getdata())
    assert logo.mode == image_mode
    assert logo.tobytes() == before
    assert group["file"] is logo
