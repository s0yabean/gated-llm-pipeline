"""Builders for classifier-shaped answers, so policy tests do not depend on any backend."""

FILTERS = ("unsafe", "sensitive", "promo", "bait", "institutional", "political", "shill",
           "two_paths", "in_motion", "stuck", "founder")


def answers(**over):
    a = {k: {"type": "noul", "noul": 0.05} for k in FILTERS}
    a["reply_room"] = {"type": "noul", "noul": 0.9}
    a["substance"] = {"type": "noul", "noul": 0.9}
    a["theme"] = {"type": "choice", "choice": "startups_business",
                  "probabilities": {"startups_business": 0.9, "none": 0.1}}
    a["life_area"] = {"type": "choice", "choice": "none", "probabilities": {"none": 1.0}}
    for k, v in over.items():
        a[k] = {"type": "noul", "noul": v} if isinstance(v, float) else v
    return a


def link(strength, kind="supports", tag="pricing/fixed-scope"):
    return [{"tag": tag, "strength": strength, "kind": kind, "p": {}}]


POST = {"url": "https://example.com/posts/1", "handle": "sample_x",
        "text": "We tried fixed pricing on client projects and the invoice arguments moved to the start."}
