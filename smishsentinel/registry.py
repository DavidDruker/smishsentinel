"""Curated organization -> official-domain registry.

Stops the investigator from declaring its own guess at an organization's
official domain. Before this module existed, ``set_official_domain`` took
whatever domain the model asserted and only enforced *consistency* against
it afterward -- a convincing but wrong domain locked in just as cleanly as a
correct one, because nothing in the pipeline ever checked whether the
declaration itself was right (see README's former "Known limitations" entry
on this).

This registry is the correctness check that was missing. For the
organizations listed in ``data/organizations.json``, the domain is resolved
here, deterministically, never taken from the model. An organization that
isn't in the registry can't be locked at all -- see ``set_official_domain``
in ``tools/evidence.py`` -- so the investigation must abstain honestly
(``insufficient_evidence``) instead of guessing at a plausible-looking
domain.

Scoped deliberately small (around fifteen organizations) for this
submission's demo rather than attempting a general-purpose entity-resolution
system -- see README.md's "Known limitations".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "organizations.json"


@dataclass(frozen=True)
class OrganizationRecord:
    """One registry entry: a verified identity, not a guess."""

    canonical_name: str
    aliases: tuple[str, ...]
    domain: str
    known_pages: dict[str, str]


def _normalize(name: str) -> str:
    """Lowercase and collapse punctuation/whitespace so 'RBC Royal Bank' and
    'rbc-royal-bank!!' resolve to the same lookup key."""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


@lru_cache(maxsize=1)
def _load() -> tuple[OrganizationRecord, ...]:
    raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    return tuple(
        OrganizationRecord(
            canonical_name=entry["canonical_name"],
            aliases=tuple(entry.get("aliases", [])),
            domain=entry["domain"],
            known_pages=dict(entry.get("known_pages", {})),
        )
        for entry in raw["organizations"]
    )


@lru_cache(maxsize=1)
def _index() -> dict[str, OrganizationRecord]:
    """Every name form (canonical name plus every alias), normalized, mapped
    to its record. Built once and cached -- the registry is a static file
    read at process start, not something that changes per-request."""
    index: dict[str, OrganizationRecord] = {}
    for record in _load():
        for name in (record.canonical_name, *record.aliases):
            index[_normalize(name)] = record
    return index


def resolve(organization: str) -> OrganizationRecord | None:
    """Look up an organization by canonical name or alias.

    Matching is case- and punctuation-insensitive but otherwise exact --
    there is no fuzzy or partial matching, because a near-match that
    resolves to the wrong organization is exactly the failure mode this
    registry exists to prevent. Returns ``None``, not a best guess, when the
    name isn't registered; the caller's job is to treat that as an honest
    "unknown," never to fall back to inventing a domain.
    """
    return _index().get(_normalize(organization or ""))


def all_organizations() -> tuple[OrganizationRecord, ...]:
    """Every registered organization, for tooling, tests, and docs."""
    return _load()
