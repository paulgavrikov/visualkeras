"""visualkeras: visualize Keras/TensorFlow models.

This package provides simple ways to render neural network architectures as
layered or graph-style diagrams.

Main entry points:
- layered_view(model, ...): Render a layered (CNN-style) diagram
- graph_view(model, ...): Render a graph-based diagram for general models
- SpacingDummyLayer(...): Add logical spacing/grouping in layered views

Quick start:
    import visualkeras
    img = visualkeras.layered_view(model)  # returns a PIL.Image
    img.show()

Dependencies: Pillow and NumPy are required. TensorFlow/Keras are needed to
construct models for visualization. See the project README for details and
advanced options.

GitHub: https://github.com/paulgavrikov/visualkeras
PyPi: https://pypi.org/project/visualkeras/
"""

from visualkeras.layered import *
from visualkeras.graph import *
from visualkeras.options import (
    LayeredOptions,
    GraphOptions,
    LAYERED_PRESETS,
    GRAPH_PRESETS,
    LAYERED_TEXT_CALLABLES,
)
from visualkeras.show import show
