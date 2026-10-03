#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""A map change marks the assets that read it stale, and no others."""

import dataclasses

import pytest

from lakehouse_cv.contract.split_roles import SplitRole
from lakehouse_cv.defs import gold, silver
from lakehouse_cv.sources.source_registry import SOURCE_BY_NAME


def _silver_versions() -> dict[str, str | None]:
    return {
        name: silver.build_silver_asset(name).get_asset_spec().code_version
        for name in SOURCE_BY_NAME
    }


def test_every_asset_declares_a_code_version() -> None:
    assert all(version for version in _silver_versions().values())


def test_the_silver_assets_do_not_share_a_code_version() -> None:
    versions = _silver_versions()
    assert len(set(versions.values())) == len(versions)


def test_a_category_map_change_marks_one_silver_asset_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _silver_versions()
    source = SOURCE_BY_NAME["wider_face"]
    changed = dataclasses.replace(source.spec, category_map={"face": "head"})
    monkeypatch.setattr(target=source, name="spec", value=changed)

    after = _silver_versions()
    assert after["wider_face"] != before["wider_face"]
    assert after["pp4av"] == before["pp4av"]
    assert after["open_images"] == before["open_images"]


def test_a_new_embedding_model_marks_every_silver_asset_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vectors are part of silver, so a model swap has to rebuild the layer."""
    before = _silver_versions()

    monkeypatch.setattr(target=silver, name="EMBEDDING_MODEL", value="mobileclip_s2")

    after = _silver_versions()
    assert all(after[name] != before[name] for name in before)


def test_a_split_role_change_marks_gold_stale_and_no_silver_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    silver_before = _silver_versions()
    gold_before = gold.build_gold_asset().get_asset_spec().code_version
    source = SOURCE_BY_NAME["wider_face"]
    changed = dataclasses.replace(
        source.spec, split_roles={"train": SplitRole.TRAIN, "val": SplitRole.TEST}
    )
    monkeypatch.setattr(target=source, name="spec", value=changed)

    assert gold.build_gold_asset().get_asset_spec().code_version != gold_before
    assert _silver_versions() == silver_before


def test_a_spec_rejects_a_split_with_no_role() -> None:
    spec = SOURCE_BY_NAME["wider_face"].spec
    with pytest.raises(expected_exception=ValueError, match="one role"):
        dataclasses.replace(spec, split_roles={"train": SplitRole.TRAIN})
