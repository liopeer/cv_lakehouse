#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Enforce the import rules of docs/adr/0003 and docs/adr/0004."""

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CORE = "lakehouse_core"
CONTRACT_THIRD_PARTY = frozenset({"pyarrow", "pydantic"})


@dataclass(frozen=True)
class Module:
    name: str
    package: str
    imports: tuple[str, ...]


def _read_module(path: Path, src_dir: Path) -> Module:
    parts = path.relative_to(src_dir).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    imports = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imports.append(node.module)
    return Module(name=".".join(parts), package=parts[0], imports=tuple(imports))


def _read_modules() -> list[Module]:
    src_dirs = [*REPO.glob("packages/*/src"), *REPO.glob("domains/*/src")]
    return [
        _read_module(path=path, src_dir=src_dir)
        for src_dir in src_dirs
        for path in sorted(src_dir.rglob("*.py"))
    ]


MODULES = _read_modules()
DOMAINS = sorted({module.package for module in MODULES} - {CORE})


def _is_within(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def test_the_modules_are_found() -> None:
    assert any(module.package == CORE for module in MODULES)
    assert DOMAINS


@pytest.mark.parametrize("module", MODULES, ids=lambda module: module.name)
def test_a_package_imports_only_itself_and_core(module: Module) -> None:
    allowed = {module.package, CORE}
    foreign = [
        name
        for name in module.imports
        if name.split(".")[0].startswith("lakehouse_")
        and name.split(".")[0] not in allowed
    ]
    assert not foreign, f"{module.name} imports {foreign}"


@pytest.mark.parametrize(
    "module",
    [module for module in MODULES if ".contract" in module.name],
    ids=lambda module: module.name,
)
def test_a_contract_imports_only_light_modules(module: Module) -> None:
    contract = f"{module.package}.contract"
    rejected = [
        name
        for name in module.imports
        if not _is_within(name, contract)
        and not _is_within(name, CORE)
        and name.split(".")[0] not in CONTRACT_THIRD_PARTY
        and name.split(".")[0] not in sys.stdlib_module_names
    ]
    assert not rejected, f"{module.name} imports {rejected}"


@pytest.mark.parametrize("module", MODULES, ids=lambda module: module.name)
def test_only_the_wiring_imports_the_definitions(module: Module) -> None:
    defs = f"{module.package}.defs"
    if _is_within(module.name, defs) or module.name == f"{module.package}.definitions":
        return
    wiring = [name for name in module.imports if _is_within(name, defs)]
    assert not wiring, f"{module.name} imports {wiring}"
