"""Package the reviewed static site using an explicit public-file list.

No application build runs here. The deployment output is always the source
repository's dedicated _site directory; symlinks and private metadata are
never copied into the Pages artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys


PUBLIC_FILES = (
    "index.html",
    "gaussian-landscape.html",
    "local-manifold.html",
    "guided-shells.html",
    "generating-by-climbing.md",
    "README.md",
    "figure-checks.json",
    ".nojekyll",
)
PUBLIC_DIRECTORIES = ("assets", "interactive", "manifold", "guidance", "licenses")
PRIVATE_DIRECTORIES = {".git", ".github", ".openai", "node_modules", "__pycache__"}
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".zip", ".7z")


def is_private(path: Path) -> bool:
    return (
        any(part in PRIVATE_DIRECTORIES for part in path.parts)
        or path.name == ".env"
        or path.name.startswith(".env.")
        or path.name in {".DS_Store", "Thumbs.db"}
        or path.name.lower().endswith(ARCHIVE_SUFFIXES)
    )


def validate_relative_browser_paths(source: Path) -> None:
    """Reject site-root URLs that would break project-subdirectory hosting."""
    documents = [source / name for name in PUBLIC_FILES if name.endswith(".html")]
    documents += [source / folder / "index.html" for folder in ("interactive", "manifold", "guidance")]
    for document in documents:
        if not document.is_file():
            continue
        for attribute, value in re.findall(r"\b(href|src|srcset)\s*=\s*[\"']([^\"']+)[\"']", document.read_text(encoding="utf-8")):
            if value.startswith("/") and not value.startswith("//"):
                relative = document.relative_to(source)
                raise ValueError(f"Project-subdirectory-incompatible {attribute} in {relative}: {value}")
    stylesheets = [path for folder in PUBLIC_DIRECTORIES for path in (source / folder).rglob("*.css")]
    for stylesheet in stylesheets:
        for value in re.findall(r"url\(\s*[\"']?([^\"')]+)", stylesheet.read_text(encoding="utf-8")):
            if value.startswith("/") and not value.startswith("//"):
                relative = stylesheet.relative_to(source)
                raise ValueError(f"Project-subdirectory-incompatible CSS URL in {relative}: {value}")


def copy_public(source: Path, output: Path) -> dict[str, object]:
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("The source must be a repository directory.")
    # Only this dedicated generated directory may be replaced. This check
    # intentionally rejects arbitrary paths, parent directories and symlinks.
    expected_output = source / "_site"
    if output.is_symlink() or output.resolve() != expected_output:
        raise ValueError("The output must be the source repository's _site directory.")
    if output.exists() and not output.is_dir():
        raise ValueError("The output path exists and is not a directory.")
    missing = [name for name in PUBLIC_FILES if not (source / name).is_file()]
    missing += [name for name in PUBLIC_DIRECTORIES if not (source / name).is_dir()]
    if missing:
        raise ValueError("Public site files are missing: " + ", ".join(missing))

    paths: list[Path] = []
    for name in PUBLIC_FILES:
        entry = source / name
        if entry.is_symlink():
            raise ValueError(f"Symlinks are not allowed in the public site: {name}")
        paths.append(entry)
    for name in PUBLIC_DIRECTORIES:
        root = source / name
        if root.is_symlink():
            raise ValueError(f"Symlinks are not allowed in the public site: {name}")
        for entry in root.rglob("*"):
            relative = entry.relative_to(source)
            if entry.is_symlink():
                raise ValueError(f"Symlinks are not allowed in the public site: {relative}")
            if entry.is_file() and not is_private(relative):
                paths.append(entry)
    validate_relative_browser_paths(source)

    # Resolved and checked above: deletion is confined to source/_site.
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for entry in paths:
        destination = output / entry.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry, destination)
    return {
        "output": str(output),
        "files": len(paths),
        "bytes": sum(entry.stat().st_size for entry in paths),
        "public_files": list(PUBLIC_FILES),
        "public_directories": list(PUBLIC_DIRECTORIES),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("_site"))
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output if args.output.is_absolute() else source / args.output
    print(json.dumps(copy_public(source, output), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Cannot package the public site: {error}", file=sys.stderr)
        raise SystemExit(1)
