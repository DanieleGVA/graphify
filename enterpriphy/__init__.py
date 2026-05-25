"""enterpriphy — extract · build · cluster · analyze · report.

v1 is the rebrand and enterprise evolution of `graphify` (PyPI `graphifyy`).
See ARCHITECTURE_v2.md and MIGRATION_PLAN.md in the repo for the target state.
"""


def __getattr__(name):
    # Lazy imports so `enterpriphy install` works before heavy deps are in place.
    _map = {
        "extract": ("enterpriphy.extract", "extract"),
        "collect_files": ("enterpriphy.extract", "collect_files"),
        "build_from_json": ("enterpriphy.build", "build_from_json"),
        "cluster": ("enterpriphy.cluster", "cluster"),
        "score_all": ("enterpriphy.cluster", "score_all"),
        "cohesion_score": ("enterpriphy.cluster", "cohesion_score"),
        "god_nodes": ("enterpriphy.analyze", "god_nodes"),
        "surprising_connections": ("enterpriphy.analyze", "surprising_connections"),
        "suggest_questions": ("enterpriphy.analyze", "suggest_questions"),
        "generate": ("enterpriphy.report", "generate"),
        "to_json": ("enterpriphy.export", "to_json"),
        "to_html": ("enterpriphy.export", "to_html"),
        "to_svg": ("enterpriphy.export", "to_svg"),
        "to_canvas": ("enterpriphy.export", "to_canvas"),
        "to_wiki": ("enterpriphy.wiki", "to_wiki"),
    }
    if name in _map:
        import importlib
        mod_name, attr = _map[name]
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module 'enterpriphy' has no attribute {name!r}")
