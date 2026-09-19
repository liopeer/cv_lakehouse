#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The class registry is frozen data. These tests guard it against a bad edit."""

from cv_lakehouse.class_registry import (
    LABELFORMAT_CATEGORIES,
    CanonicalClass,
    class_registry_sha256_fingerprint,
    sha256_fingerprint,
)

# These names and ids are written down. They never change.
KNOWN = {
    "face": 0,
    "license_plate": 1,
    "head": 2,
    "person": 3,
    "text": 4,
    "other": 5,
}


def test_the_registry_holds_the_classes_we_need() -> None:
    assert [member.class_name for member in CanonicalClass] == list(KNOWN)


def test_the_known_classes_keep_their_ids() -> None:
    for class_name, class_id in KNOWN.items():
        assert CanonicalClass.from_class_name(class_name).value == class_id


def test_the_ids_are_contiguous() -> None:
    ids = sorted(member.value for member in CanonicalClass)
    assert ids == list(range(len(ids)))


def test_every_name_is_a_slug() -> None:
    for member in CanonicalClass:
        assert member.class_name.replace("_", "").isalnum()
        assert member.class_name.islower()


def test_every_name_is_listed_and_looks_itself_up() -> None:
    assert CanonicalClass.all_class_names() == set(KNOWN)
    for class_name in CanonicalClass.all_class_names():
        assert CanonicalClass.from_class_name(class_name).class_name == class_name


def test_the_categories_mirror_the_classes() -> None:
    assert [(c.id, c.name) for c in LABELFORMAT_CATEGORIES] == [
        (member.value, member.class_name) for member in CanonicalClass
    ]


def test_the_registry_fingerprint_is_stable_and_short() -> None:
    assert class_registry_sha256_fingerprint() == class_registry_sha256_fingerprint()
    assert len(class_registry_sha256_fingerprint()) == 12


def test_a_fingerprint_depends_on_every_part() -> None:
    assert sha256_fingerprint(["a", "b"]) == sha256_fingerprint(["a", "b"])
    assert sha256_fingerprint(["a", "b"]) != sha256_fingerprint(["a", "c"])
    # A separator stops two different splits colliding.
    assert sha256_fingerprint(["a", "bc"]) != sha256_fingerprint(["ab", "c"])
