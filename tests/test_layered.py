from visualkeras.layered import layered_view


def test_graph_view(model):
    layered_view(model)