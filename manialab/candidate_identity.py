import re
import uuid


_IDENTITY = re.compile(r"^[0-9a-f]{32}$")


def new_candidate_id():
    return uuid.uuid4().hex


def ensure_candidate_ids(state):
    """Assign unique persistent identities to candidates missing them."""
    changed = False
    seen = set()
    variants = state.get("variants", [])
    if not isinstance(variants, list):
        return False

    for variant in variants:
        if not isinstance(variant, dict):
            continue
        candidate_id = variant.get("candidate_id")
        if not isinstance(candidate_id, str) or not _IDENTITY.fullmatch(candidate_id) or candidate_id in seen:
            candidate_id = new_candidate_id()
            while candidate_id in seen:
                candidate_id = new_candidate_id()
            variant["candidate_id"] = candidate_id
            changed = True
        seen.add(candidate_id)
    return changed
