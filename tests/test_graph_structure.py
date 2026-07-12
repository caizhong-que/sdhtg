import torch

from sdhtg.models.sdhtg import SDHTG
from tests.model_helpers import make_batch, tiny_config


EXPECTED_RELATIONS = {
    ("status", "temporal", "status"),
    ("status", "semantic", "status"),
    ("action", "temporal", "action"),
    ("action", "semantic", "action"),
    ("entity", "temporal", "entity"),
    ("entity", "semantic", "entity"),
    ("status", "belongs_to", "action"),
    ("action", "contains", "status"),
    ("action", "belongs_to", "entity"),
    ("entity", "contains", "action"),
}


def test_graph_contains_all_declared_node_and_relation_types():
    model = SDHTG(tiny_config()).eval()
    output = model(make_batch(lengths=(6, 3), total_steps=6))
    node_types, edge_types = output.graph_batch.metadata()
    assert set(node_types) == {"status", "action", "entity"}
    assert set(edge_types) == EXPECTED_RELATIONS

    for relation in edge_types:
        store = output.graph_batch[relation]
        assert store.edge_index.shape[0] == 2
        assert store.edge_attr.shape[1] == 2
        assert store.edge_index.shape[1] == store.edge_attr.shape[0]


def test_reverse_containment_uses_reversed_indices():
    model = SDHTG(tiny_config()).eval()
    output = model(make_batch(lengths=(6,), total_steps=6))
    graph = output.graph_batch
    forward = graph[("status", "belongs_to", "action")].edge_index
    reverse = graph[("action", "contains", "status")].edge_index
    assert torch.equal(forward.flip(0), reverse)
