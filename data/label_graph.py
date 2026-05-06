"""Label hierarchy graph construction for schema-aware classification."""

import torch


def _ag_news_edges():
    """AG News: 4 classes — World(0), Sports(1), Business(2), Sci/Tech(3)."""
    edges = [
        (0, 1), (1, 0),  # World <-> Sports
        (0, 2), (2, 0),  # World <-> Business
        (2, 3), (3, 2),  # Business <-> Sci/Tech
    ]
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    return torch.tensor([src, dst], dtype=torch.long)


def _imdb_edges():
    """IMDB: 2 classes — Negative(0), Positive(1)."""
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.long)


def _20newsgroups_edges():
    """20 Newsgroups: 20 classes with natural hierarchy.

    Groups:
        comp.*   : 0-4   (graphics, os.ms-windows, sys.ibm, sys.mac, windows.x)
        rec.*    : 5-8   (autos, motorcycles, sport.baseball, sport.hockey)
        sci.*    : 9-12  (crypt, electronics, med, space)
        talk.*   : 13-16 (politics.misc, politics.guns, politics.mideast, religion.misc)
        misc     : 17    (misc.forsale)
        alt      : 18    (alt.atheism)
        soc      : 19    (soc.religion.christian)
    """
    edges = []

    # Fully connect within comp.* (0-4)
    comp = [0, 1, 2, 3, 4]
    for i in comp:
        for j in comp:
            if i != j:
                edges.append((i, j))

    # Fully connect within rec.* (5-8)
    rec = [5, 6, 7, 8]
    for i in rec:
        for j in rec:
            if i != j:
                edges.append((i, j))

    # Fully connect within sci.* (9-12)
    sci = [9, 10, 11, 12]
    for i in sci:
        for j in sci:
            if i != j:
                edges.append((i, j))

    # Fully connect within talk.* (13-16)
    talk = [13, 14, 15, 16]
    for i in talk:
        for j in talk:
            if i != j:
                edges.append((i, j))

    # Cross-group edges
    # religion cluster: talk.religion.misc(16) <-> alt.atheism(18) <-> soc.religion.christian(19)
    for a, b in [(16, 18), (16, 19), (18, 19)]:
        edges.append((a, b))
        edges.append((b, a))

    # sci <-> comp (scientific computing)
    for s in [9, 10]:
        for c in [0, 4]:
            edges.append((s, c))
            edges.append((c, s))

    # rec.sport <-> misc.forsale (sports equipment)
    for r in [7, 8]:
        edges.append((r, 17))
        edges.append((17, r))

    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    return torch.tensor([src, dst], dtype=torch.long)


def _bbc_edges():
    """BBC News: 5 classes — business(0), entertainment(1), politics(2), sport(3), tech(4)."""
    edges = [
        (0, 4), (4, 0),  # business <-> tech
        (0, 2), (2, 0),  # business <-> politics
        (1, 2), (2, 1),  # entertainment <-> politics
    ]
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    return torch.tensor([src, dst], dtype=torch.long)


def build_label_graph(dataset_name: str):
    """Build the label graph edge_index for a dataset.

    Returns:
        edge_index: Tensor of shape (2, num_edges) with bidirectional edges.
    """
    builders = {
        "ag_news": _ag_news_edges,
        "imdb": _imdb_edges,
        "20newsgroups": _20newsgroups_edges,
        "bbc": _bbc_edges,
    }

    if dataset_name not in builders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(builders.keys())}")

    edge_index = builders[dataset_name]()
    print(f"Label graph for {dataset_name}: {edge_index.shape[1]} edges")
    return edge_index
