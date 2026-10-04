"""
[INPUT] CPubMed-KG 2.0 triples and the fixed ICD-only candidate catalog.
[OUTPUT] Disease entities, exact ICD alignments, direct synonyms and ICD graph edges.
[POS] Reproducible knowledge-graph evidence layer for the unified Phase 1 ranker.
[UPDATE] Keep parsing, alias ambiguity and graph-edge policies synchronized with SPEC_PHASE1.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mednorm.data import sha256_file
from mednorm.text import normalize_label, normalize_text


@dataclass(frozen=True, slots=True)
class KGEntity:
    name: str
    entity_type: str | None


@dataclass(frozen=True, slots=True)
class KGTriple:
    head: KGEntity
    relation: str
    tail: KGEntity


def parse_entity(value: str) -> KGEntity:
    """Parse CPubMed-KG entities; synonym rows intentionally omit type suffixes."""
    normalized = normalize_text(value)
    if not normalized:
        raise ValueError("KG entity must not be empty")
    if "@@" not in normalized:
        return KGEntity(normalize_label(normalized), None)
    name, entity_type = normalized.rsplit("@@", 1)
    name = normalize_label(name)
    entity_type = normalize_text(entity_type)
    if not name or not entity_type:
        raise ValueError(f"invalid typed KG entity: {value!r}")
    return KGEntity(name, entity_type)


def parse_triple(line: str, *, line_number: int) -> KGTriple:
    columns = line.rstrip("\r\n").split("\t")
    if len(columns) != 3:
        raise ValueError(
            f"CPubMed-KG line {line_number} must have exactly three tab-separated columns"
        )
    relation = normalize_text(columns[1])
    if not relation:
        raise ValueError(f"CPubMed-KG line {line_number} has an empty relation")
    return KGTriple(parse_entity(columns[0]), relation, parse_entity(columns[2]))


def _read_catalog(path: Path) -> tuple[str, ...]:
    names: list[str] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("name"), str):
                    raise ValueError(f"catalog line {line_number} has invalid name")
                name = normalize_label(row["name"])
                if not name:
                    raise ValueError(f"catalog line {line_number} has empty name")
                names.append(name)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid ICD catalog {path}: {exc}") from exc
    if not names:
        raise ValueError("ICD catalog must contain at least one name")
    if len(names) != len(set(names)):
        raise ValueError("ICD catalog contains duplicate names")
    return tuple(names)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_kg_index(
    *, kg_path: Path, catalog_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Build a deterministic, local-only CPubMed-KG index for ICD candidates."""
    if not kg_path.is_file():
        raise FileNotFoundError(f"CPubMed-KG file not found: {kg_path}")
    candidate_names = _read_catalog(catalog_path)
    candidate_set = set(candidate_names)
    candidate_order = {name: index for index, name in enumerate(candidate_names)}

    disease_entities: set[str] = set()
    synonym_nodes: set[str] = set()
    direct_alias_targets: dict[str, set[str]] = {}
    icd_edges: set[tuple[str, str, str]] = set()
    relation_counts: Counter[str] = Counter()
    endpoint_type_counts: Counter[str] = Counter()
    triple_count = 0
    synonym_triple_count = 0
    untyped_endpoint_count = 0

    with kg_path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\r\n")
        if header != "head_entity@@entity_type\trelation\ttail_entity@@entity_type":
            raise ValueError(f"unexpected CPubMed-KG header: {header!r}")
        for line_number, line in enumerate(handle, start=2):
            triple = parse_triple(line, line_number=line_number)
            triple_count += 1
            relation_counts[triple.relation] += 1
            for entity in (triple.head, triple.tail):
                if entity.entity_type is None:
                    untyped_endpoint_count += 1
                else:
                    endpoint_type_counts[entity.entity_type] += 1
                    if entity.entity_type == "疾病":
                        disease_entities.add(entity.name)

            if triple.relation == "同义词":
                synonym_triple_count += 1
                synonym_nodes.add(triple.head.name)
                synonym_nodes.add(triple.tail.name)
                if (
                    triple.head.name in candidate_set
                    and triple.tail.name not in candidate_set
                ):
                    direct_alias_targets.setdefault(triple.tail.name, set()).add(
                        triple.head.name
                    )
                if (
                    triple.tail.name in candidate_set
                    and triple.head.name not in candidate_set
                ):
                    direct_alias_targets.setdefault(triple.head.name, set()).add(
                        triple.tail.name
                    )
                if (
                    triple.head.name in candidate_set
                    and triple.tail.name in candidate_set
                    and triple.head.name != triple.tail.name
                ):
                    icd_edges.add((triple.head.name, triple.relation, triple.tail.name))

            if (
                triple.head.entity_type == "疾病"
                and triple.tail.entity_type == "疾病"
                and triple.head.name in candidate_set
                and triple.tail.name in candidate_set
            ):
                icd_edges.add((triple.head.name, triple.relation, triple.tail.name))

    exact_alignments = tuple(name for name in candidate_names if name in disease_entities)

    alias_rows: list[dict[str, Any]] = []
    ambiguous_alias_count = 0
    for alias in sorted(direct_alias_targets):
        ordered_anchors = sorted(
            direct_alias_targets[alias], key=candidate_order.__getitem__
        )
        ambiguous_alias_count += int(len(ordered_anchors) > 1)
        alias_rows.append(
            {
                "alias": alias,
                "icd_names": ordered_anchors,
                "typed_as_disease": alias in disease_entities,
                "evidence": "direct_synonym_edge",
            }
        )

    ordered_edges = sorted(
        icd_edges,
        key=lambda edge: (
            candidate_order[edge[0]],
            edge[1],
            candidate_order[edge[2]],
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    disease_path = output_dir / "disease_entities.txt"
    disease_path.write_text(
        "".join(f"{name}\n" for name in sorted(disease_entities)), encoding="utf-8"
    )
    _write_jsonl(
        output_dir / "icd_alignment.jsonl",
        ({"icd_name": name} for name in exact_alignments),
    )
    _write_jsonl(output_dir / "aliases.jsonl", alias_rows)
    _write_jsonl(
        output_dir / "icd_edges.jsonl",
        (
            {"source": source, "relation": relation, "target": target}
            for source, relation, target in ordered_edges
        ),
    )

    stats: dict[str, Any] = {
        "spec_id": "MEDNORM-P1-001",
        "triple_count": triple_count,
        "relation_count": len(relation_counts),
        "relation_counts": dict(sorted(relation_counts.items())),
        "endpoint_type_counts": dict(sorted(endpoint_type_counts.items())),
        "untyped_endpoint_count": untyped_endpoint_count,
        "synonym_triple_count": synonym_triple_count,
        "synonym_node_count": len(synonym_nodes),
        "disease_entity_count": len(disease_entities),
        "icd_candidate_count": len(candidate_names),
        "exact_icd_disease_alignment_count": len(exact_alignments),
        "alias_count": len(alias_rows),
        "ambiguous_alias_count": ambiguous_alias_count,
        "alias_policy": (
            "one-hop only: transitive synonym closure is forbidden because overloaded "
            "abbreviations merge unrelated diseases"
        ),
        "icd_edge_count": len(ordered_edges),
        "input_sha256": {
            "cpubmed_kg": sha256_file(kg_path),
            "icd_catalog": sha256_file(catalog_path),
        },
        "license_note": "CPubMed-KG 2.0 is CC BY-ND 4.0; generated index stays local.",
    }
    _write_json(output_dir / "stats.json", stats)
    return stats
