import faiss
import numpy as np
import pytest

from faissight.core import hnsw as H
from faissight.core.loader import load_index
from tests.conftest import SMALL

N, M = SMALL["n"], SMALL["hnsw_m"]


@pytest.fixture(scope="module")
def graph(synthetic):
    li = load_index(synthetic["hnsw_flat"])
    return li, H.extract_graph(li)


def test_graph_shape(graph) -> None:
    li, g = graph
    assert g.ntotal == N
    assert g.max_level == li.params.max_level
    assert g.entry_point == li.params.entry_point
    assert g.node_levels[g.entry_point] == g.max_level
    assert g.node_levels.min() == 0
    assert g.node_levels.max() == g.max_level


def test_levels_shrink_upwards(graph) -> None:
    _, g = graph
    counts = [len(g.nodes_at_level(level)) for level in range(g.max_level + 1)]
    assert counts[0] == N
    assert counts == sorted(counts, reverse=True)
    assert g.entry_point in g.nodes_at_level(g.max_level)


def test_links_point_to_nodes_on_the_same_level(graph) -> None:
    _, g = graph
    for level in range(g.max_level + 1):
        cap = 2 * M if level == 0 else M
        for node in g.nodes_at_level(level)[:200]:
            nbrs = g.neighbours(int(node), level)
            assert len(nbrs) <= cap
            assert (g.node_levels[nbrs] >= level).all()
            assert node not in nbrs


def test_neighbours_above_node_level_empty(graph) -> None:
    _, g = graph
    ground = int(np.flatnonzero(g.node_levels == 0)[0])
    assert len(g.neighbours(ground, 1)) == 0


def test_level0_links_mostly_close(graph, synthetic) -> None:
    # HNSW links connect near neighbours: linked pairs are far closer than random pairs.
    _, g = graph
    x = np.load(synthetic["vectors"])
    src, dst = g.edges(0)
    linked = np.square(x[src] - x[dst]).sum(1).mean()
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, N, 2000), rng.integers(0, N, 2000)
    assert linked < 0.5 * np.square(x[a] - x[b]).sum(1).mean()


def test_edges_match_neighbours(graph) -> None:
    _, g = graph
    level = 1
    src, dst = g.edges(level)
    for node in g.nodes_at_level(level)[:50]:
        np.testing.assert_array_equal(
            np.sort(dst[src == node]), np.sort(g.neighbours(int(node), level))
        )


def test_edges_restricted_to_subset(graph) -> None:
    _, g = graph
    subset = np.arange(300)
    src, dst = g.edges(0, subset)
    assert np.isin(src, subset).all()
    assert np.isin(dst, subset).all()


def test_degrees_and_stats(graph) -> None:
    _, g = graph
    st = H.graph_stats(g)
    assert st.entry_point == g.entry_point
    assert [s.level for s in st.levels] == list(range(g.max_level, -1, -1))
    for s in st.levels:
        assert s.max_links == (2 * M if s.level == 0 else M)
        assert sum(s.degree_hist) == s.n_nodes
        assert len(s.degree_hist) == s.max_links + 1
        assert s.degree_min <= s.degree_mean <= s.degree_max <= s.max_links
    level0 = st.levels[-1]
    assert level0.n_nodes == N
    assert level0.degree_min >= 1  # every node reachable from somewhere


def test_neighbourhood_bfs(graph) -> None:
    _, g = graph
    nodes = H.neighbourhood(g, [5], 0, 100)
    assert nodes[0] == 5
    assert len(nodes) == 100
    assert len(set(nodes.tolist())) == 100
    first_ring = set(g.neighbours(5, 0).tolist())
    assert first_ring <= set(nodes.tolist())
    # Seeds not on the level are skipped.
    ground = int(np.flatnonzero(g.node_levels == 0)[0])
    assert len(H.neighbourhood(g, [ground], 1, 10)) == 0


def test_idmap_and_other_storage() -> None:
    x = np.random.default_rng(0).standard_normal((800, 16)).astype(np.float32)
    idmap = faiss.IndexIDMap(faiss.IndexHNSWFlat(16, 8))
    idmap.add_with_ids(x, np.arange(800) * 3)
    assert H.extract_graph(load_index(idmap)).ntotal == 800
    sq = faiss.IndexHNSWSQ(16, faiss.ScalarQuantizer.QT_8bit, 8)
    sq.train(x)
    sq.add(x)
    g = H.extract_graph(load_index(sq))
    assert g.ntotal == 800
    assert len(g.neighbours(g.entry_point, g.max_level)) > 0 or g.max_level == 0


def test_not_hnsw(synthetic) -> None:
    with pytest.raises(H.NotAnHnswIndexError):
        H.extract_graph(load_index(synthetic["ivf_flat"]))


def test_degrees_are_the_same_in_small_batches(graph, monkeypatch) -> None:
    _, g = graph
    expected = [g.degrees(level) for level in range(g.max_level + 1)]
    monkeypatch.setattr(H, "_BATCH_CELLS", 7)  # a few nodes per batch, with a ragged tail
    for level, want in enumerate(expected):
        np.testing.assert_array_equal(g.degrees(level), want)
    assert g.degrees(0).sum() == len(g.edges(0)[0])


def test_graph_borrows_the_index_links_read_only() -> None:
    x = np.random.default_rng(0).standard_normal((200, 8)).astype(np.float32)
    index = faiss.IndexHNSWFlat(8, 4)
    index.add(x)
    g = H.extract_graph(load_index(index))
    expected = faiss.vector_to_array(index.hnsw.neighbors)
    del index  # the graph keeps the index (and so the borrowed memory) alive
    np.testing.assert_array_equal(g.neighbors, expected)
    assert not g.neighbors.flags.writeable
    assert not g.offsets.flags.writeable
    empty = H.extract_graph(load_index(faiss.IndexHNSWFlat(8, 4)))
    assert empty.ntotal == 0
    assert len(empty.neighbors) == 0
