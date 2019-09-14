from collections import defaultdict

import numpy as np

from .chuliu_edmonds import chuliu_edmonds_one_root


def mst(arc_probs, rel_probs=None, use_chi_liu_edmonds=True):
    if use_chi_liu_edmonds:
        arcs = chuliu_edmonds_one_root(arc_probs)
    else:
        arcs = _arc_argmax(arc_probs)
    arcs[0] = 0
    if rel_probs is not None:
        rels = _rel_argmax(rel_probs[np.arange(len(arcs)), arcs])
        rels[0] = -1
    else:
        rels = None
    arcs[0] = -1
    return arcs, rels


def _arc_argmax(probs):
    """
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/models/nn.py#L532  # NOQA
    """
    length = probs.shape[0]
    probs = probs * (1 - np.eye(length))
    heads = np.argmax(probs, axis=1)
    tokens = np.arange(1, length)
    roots = np.where(heads[tokens] == 0)[0] + 1
    if len(roots) < 1:
        root_probs = probs[tokens, 0]
        head_probs = probs[tokens, heads[tokens]]
        new_root = tokens[np.argmax(root_probs / head_probs)]
        heads[new_root] = 0
    elif len(roots) > 1:
        root_probs = probs[roots, 0]
        probs[roots, 0] = 0
        new_heads = np.argmax(probs[roots][:, tokens], axis=1) + 1
        new_root = roots[np.argmin(probs[roots, new_heads] / root_probs)]
        heads[roots] = new_heads
        heads[new_root] = 0
    edges = defaultdict(set)
    vertices = set((0,))
    for dep, head in enumerate(heads[tokens]):
        vertices.add(dep + 1)
        edges[head].add(dep + 1)
    for cycle in _find_cycle(vertices, edges):
        dependents = set()
        to_visit = set(cycle)
        while len(to_visit) > 0:
            node = to_visit.pop()
            if node not in dependents:
                dependents.add(node)
                to_visit.update(edges[node])
        cycle = np.array(list(cycle))
        old_heads = heads[cycle]
        old_probs = probs[cycle, old_heads]
        non_heads = np.array(list(dependents))
        probs[np.repeat(cycle, len(non_heads)),
              np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
        new_heads = np.argmax(probs[cycle][:, tokens], axis=1) + 1
        new_probs = probs[cycle, new_heads] / old_probs
        change = np.argmax(new_probs)
        changed_cycle = cycle[change]
        old_head = old_heads[change]
        new_head = new_heads[change]
        heads[changed_cycle] = new_head
        edges[new_head].add(changed_cycle)
        edges[old_head].remove(changed_cycle)
    return heads


def _rel_argmax(probs, root_rel=0):
    """
    https://github.com/tdozat/Parser-v1/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/models/nn.py#L612  # NOQA
    """
    length = probs.shape[0]
    tokens = np.arange(1, length)
    rels = np.argmax(probs, axis=1)
    roots = np.where(rels[tokens] == root_rel)[0] + 1
    if len(roots) < 1:
        rels[1 + np.argmax(probs[tokens, root_rel])] = root_rel
    elif len(roots) > 1:
        root_probs = probs[roots, root_rel]
        probs[roots, root_rel] = 0
        new_rels = np.argmax(probs[roots], axis=1)
        new_probs = probs[roots, new_rels] / root_probs
        new_root = roots[np.argmin(new_probs)]
        rels[roots] = new_rels
        rels[new_root] = root_rel
    return rels


def _find_cycle(vertices, edges):
    """
    https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm  # NOQA
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/etc/tarjan.py  # NOQA
    """
    index = 0
    stack = []
    indices = {}
    lowlinks = {}
    onstack = defaultdict(lambda: False)
    SCCs = []

    def strongconnect(v):
        nonlocal index
        indices[v] = index
        lowlinks[v] = index
        index += 1
        stack.append(v)
        onstack[v] = True

        for w in edges[v]:
            if w not in indices:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif onstack[w]:
                lowlinks[v] = min(lowlinks[v], indices[w])

        if lowlinks[v] == indices[v]:
            SCC = set()
            while True:
                w = stack.pop()
                onstack[w] = False
                SCC.add(w)
                if not(w != v):
                    break
            SCCs.append(SCC)

    for v in vertices:
        if v not in indices:
            strongconnect(v)

    return [SCC for SCC in SCCs if len(SCC) > 1]
