from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vaxreplay.agentic import firecracker_qualification_runtime_closure as runtime_closure


def _write_fake_interpreter(path: Path, *, prefix: Path, import_paths: tuple[Path, ...]) -> None:
    payload = json.dumps(
        {
            'implementation': 'cpython',
            'version': '3.12.9',
            'executable': str(path),
            'prefix': str(prefix),
            'base_prefix': str(prefix),
            'path': [str(item) for item in import_paths],
        },
        sort_keys=True,
        separators=(',', ':'),
    )
    path.write_text(
        f'#!/bin/sh\ntest "$1" = -I || exit 91\ntest "$2" = -B || exit 92\nprintf \'%s\\n\' \'{payload}\'\n'
    )
    path.chmod(0o755)


def _build(tmp_path: Path):
    root = tmp_path / 'runtime'
    root.mkdir(mode=0o755)
    interpreter = root / 'python3'
    driver = root / 'vaxreplay-firecracker-qualification-driver'
    driver.write_bytes(f'#!{interpreter}\nprint("driver")\n'.encode())
    driver.chmod(0o755)
    package = root / 'site-packages'
    package.mkdir(mode=0o755)
    (package / 'vaxreplay.py').write_bytes(b'VALUE = 1\n')
    missing_zip = root / 'python312.zip'
    _write_fake_interpreter(interpreter, prefix=root, import_paths=(package, missing_zip))
    loaded = runtime_closure.build_and_retain_qualification_driver_runtime_closure(
        closure_id='qualification-driver-test-v1',
        driver_entrypoint_path=driver,
        interpreter_path=interpreter,
        runtime_roots=(root,),
        output_root=tmp_path / 'closure',
        source_date_epoch=1_700_000_000,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )
    return root, interpreter, driver, loaded


def test_runtime_closure_is_create_once_and_offline_verifiable(tmp_path: Path) -> None:
    root, interpreter, driver, loaded = _build(tmp_path)
    assert loaded.manifest.driver_entrypoint_path == str(driver)
    assert loaded.manifest.interpreter_path == str(interpreter)
    assert loaded.manifest.complete_declared_import_roots_inventoried is True
    assert loaded.manifest.self_contained_executable_claimed is False
    assert loaded.manifest.reproducible_build_claimed is False
    reloaded = runtime_closure.verify_qualification_driver_runtime_closure(
        Path(loaded.root),
        expected_manifest_sha256=loaded.manifest_sha256,
        expected_receipt_sha256=loaded.receipt_sha256,
        expected_closure_sha256=loaded.closure_sha256,
        require_root_owned=False,
    )
    assert reloaded == loaded
    assert any(entry.path == str(root / 'site-packages' / 'vaxreplay.py') for entry in loaded.manifest.entries)
    with pytest.raises(runtime_closure.QualificationDriverRuntimeClosureError, match='already exists'):
        runtime_closure.build_and_retain_qualification_driver_runtime_closure(
            closure_id='qualification-driver-test-v1',
            driver_entrypoint_path=driver,
            interpreter_path=interpreter,
            runtime_roots=(root,),
            output_root=Path(loaded.root),
            source_date_epoch=1_700_000_000,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
        )


def test_runtime_closure_rejects_interpreter_that_mutates_tree_during_observation(tmp_path: Path) -> None:
    root = tmp_path / 'runtime'
    root.mkdir(mode=0o755)
    interpreter = root / 'python3'
    driver = root / 'driver'
    package = root / 'site-packages'
    package.mkdir(mode=0o755)
    payload = json.dumps(
        {
            'implementation': 'cpython',
            'version': '3.12.9',
            'executable': str(interpreter),
            'prefix': str(root),
            'base_prefix': str(root),
            'path': [str(package)],
        },
        sort_keys=True,
        separators=(',', ':'),
    )
    interpreter.write_text(
        '#!/bin/sh\n'
        'test "$1" = -I || exit 91\n'
        'test "$2" = -B || exit 92\n'
        f"touch '{root / 'unexpected-cache'}'\n"
        f"printf '%s\\n' '{payload}'\n"
    )
    interpreter.chmod(0o755)
    driver.write_bytes(f'#!{interpreter}\n'.encode())
    driver.chmod(0o755)

    with pytest.raises(
        runtime_closure.QualificationDriverRuntimeClosureError,
        match='mutated its installed tree',
    ):
        runtime_closure.build_and_retain_qualification_driver_runtime_closure(
            closure_id='mutating-interpreter-v1',
            driver_entrypoint_path=driver,
            interpreter_path=interpreter,
            runtime_roots=(root,),
            output_root=tmp_path / 'closure',
            source_date_epoch=1_700_000_000,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
        )


@pytest.mark.parametrize('mutation', ['change', 'add', 'remove_import_guard'])
def test_runtime_closure_rejects_any_tree_or_search_path_change(tmp_path: Path, mutation: str) -> None:
    root, _, _, loaded = _build(tmp_path)
    if mutation == 'change':
        (root / 'site-packages' / 'vaxreplay.py').write_bytes(b'VALUE = 2\n')
    elif mutation == 'add':
        (root / 'site-packages' / 'future.py').write_bytes(b'future = True\n')
    else:
        (root / 'python312.zip').write_bytes(b'future import archive')
    with pytest.raises(runtime_closure.QualificationDriverRuntimeClosureError):
        runtime_closure.verify_qualification_driver_runtime_closure(
            Path(loaded.root),
            expected_manifest_sha256=loaded.manifest_sha256,
            expected_receipt_sha256=loaded.receipt_sha256,
            expected_closure_sha256=loaded.closure_sha256,
            require_root_owned=False,
        )


def test_runtime_closure_rejects_symlinks_and_hardlinks(tmp_path: Path) -> None:
    for kind in ('symlink', 'hardlink'):
        case = tmp_path / kind
        case.mkdir()
        root = case / 'runtime'
        root.mkdir()
        interpreter = root / 'python3'
        _write_fake_interpreter(interpreter, prefix=root, import_paths=(root,))
        driver = root / 'driver'
        driver.write_bytes(f'#!{interpreter}\n'.encode())
        driver.chmod(0o755)
        if kind == 'symlink':
            (root / 'alias').symlink_to(driver)
        else:
            os.link(driver, root / 'alias')
        with pytest.raises(runtime_closure.QualificationDriverRuntimeClosureError, match='symbolic|hardlinked'):
            runtime_closure.build_and_retain_qualification_driver_runtime_closure(
                closure_id=f'test-{kind}',
                driver_entrypoint_path=driver,
                interpreter_path=interpreter,
                runtime_roots=(root,),
                output_root=case / 'closure',
                source_date_epoch=1_700_000_000,
                expected_uid=os.getuid(),
                expected_gid=os.getgid(),
            )


def test_production_verification_rejects_non_root_owned_development_closure(tmp_path: Path) -> None:
    _, _, _, loaded = _build(tmp_path)
    if os.getuid() == 0 and os.getgid() == 0:
        pytest.skip('test needs an unprivileged fixture owner')
    with pytest.raises(runtime_closure.QualificationDriverRuntimeClosureError, match='root-owned|root ownership'):
        runtime_closure.verify_qualification_driver_runtime_closure(
            Path(loaded.root),
            expected_manifest_sha256=loaded.manifest_sha256,
            expected_receipt_sha256=loaded.receipt_sha256,
            expected_closure_sha256=loaded.closure_sha256,
            require_root_owned=True,
        )


@pytest.mark.parametrize(
    ('suffix', 'accepted'),
    [
        ('', True),
        (' -IB', True),
        (' -I -B', False),
        (' -S', False),
        (' -IB ', False),
        (' -IB -S', False),
    ],
)
def test_runtime_closure_accepts_only_exact_supported_shebang_forms(
    tmp_path: Path,
    suffix: str,
    accepted: bool,
) -> None:
    interpreter = tmp_path / 'python3'
    interpreter.write_bytes(b'python')
    entrypoint = tmp_path / 'driver'
    entrypoint.write_bytes(f'#!{interpreter}{suffix}\nprint("driver")\n'.encode())

    assert runtime_closure._entry_has_exact_shebang(entrypoint, interpreter) is accepted
