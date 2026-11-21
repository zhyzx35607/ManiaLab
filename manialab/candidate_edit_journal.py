"""Persist shared candidate edit history without repeating parent prefixes."""
import hashlib
import json


_JOURNAL = "_candidate_edit_journal"
_HEAD = "_edit_journal_head"


def _event_id(parent, event):
    encoded = json.dumps(
        {"parent": parent, "event": event},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _append(journal, parent, event):
    event_id = _event_id(parent, event)
    node = {"parent": parent, "event": event}
    previous = journal.setdefault(event_id, node)
    if previous != node:
        raise ValueError("候选编辑历史身份冲突")
    return event_id


def _events(journal, head):
    result = []
    visited = set()
    while head is not None:
        if head in visited or head not in journal:
            raise ValueError("候选编辑历史链损坏")
        visited.add(head)
        node = journal[head]
        if not isinstance(node, dict) or "event" not in node:
            raise ValueError("候选编辑历史链损坏")
        result.append(node["event"])
        head = node.get("parent")
    result.reverse()
    return result


def _reachable_journal(journal, heads):
    reachable = set()
    for head in heads:
        while head is not None and head not in reachable:
            node = journal.get(head)
            if not isinstance(node, dict):
                raise ValueError("候选编辑历史链损坏")
            reachable.add(head)
            head = node.get("parent")
    return {event_id: journal[event_id] for event_id in reachable}


def pack_state(state):
    """Store each edit event once and retain the public full-chain projection."""
    packed = dict(state)
    journal = dict(state.get(_JOURNAL, {}))
    variants = []
    heads = []

    for source in state.get("variants", []):
        variant = dict(source)
        edits = variant.get("edits", [])
        head = variant.get(_HEAD)
        if head is not None:
            inherited = _events(journal, head)
            if edits[:len(inherited)] == inherited:
                additions = edits[len(inherited):]
            else:
                # Preserve externally changed or legacy projections as a new branch.
                head = None
                additions = edits
        else:
            additions = edits

        for event in additions:
            if not isinstance(event, dict):
                raise ValueError("候选编辑历史格式无效")
            head = _append(journal, head, event)

        variant.pop("edits", None)
        if head is None:
            variant.pop(_HEAD, None)
        else:
            variant[_HEAD] = head
            heads.append(head)
        variants.append(variant)

    packed["variants"] = variants
    packed[_JOURNAL] = _reachable_journal(journal, heads)
    return packed


def unpack_state(state):
    """Restore the existing candidate.edits shape for application code and APIs."""
    journal = state.get(_JOURNAL, {})
    if not isinstance(journal, dict):
        raise ValueError("候选编辑历史格式无效")
    for variant in state.get("variants", []):
        head = variant.get(_HEAD)
        if head is not None:
            variant["edits"] = _events(journal, head)
        else:
            variant.setdefault("edits", [])
    return state


def public_state(state):
    """Strip storage-only journal metadata while retaining full edit histories."""
    result = dict(state)
    result.pop(_JOURNAL, None)
    result["variants"] = []
    for source in state.get("variants", []):
        variant = dict(source)
        variant.pop(_HEAD, None)
        result["variants"].append(variant)
    return result
