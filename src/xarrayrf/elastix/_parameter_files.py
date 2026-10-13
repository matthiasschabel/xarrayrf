"""elastix parameter files: ``(Key value ...)`` entries, read without an ITK runtime."""

from __future__ import annotations

import os
from pathlib import Path

type ParameterMap = dict[str, tuple[str, ...]]

NO_INITIAL_TRANSFORM = "NoInitialTransform"
"""The ``InitialTransformParameterFileName`` value that ends a stage chain."""


def _tokens(text: str, path: str) -> list[tuple[str, int]]:
    """Split into ``(``, ``)``, quoted strings (marked by a leading quote) and bare words.

    ``//`` starts a comment only outside quotes, so a quoted path may contain it.
    """
    tokens: list[tuple[str, int]] = []
    position, line = 0, 1
    while position < len(text):
        char = text[position]
        if char == "\n":
            line += 1
            position += 1
        elif char.isspace():
            position += 1
        elif text.startswith("//", position):
            end = text.find("\n", position)
            position = len(text) if end < 0 else end
        elif char in "()":
            tokens.append((char, line))
            position += 1
        elif char == '"':
            end = text.find('"', position + 1)
            if end < 0 or "\n" in text[position:end]:
                raise ValueError(f"{path}:{line}: unterminated quoted value")
            tokens.append((text[position:end], line))  # keeps the opening quote as a marker
            position = end + 1
        else:
            end = position
            while end < len(text) and not text[end].isspace() and text[end] not in '()"':
                if text.startswith("//", end):
                    break
                end += 1
            tokens.append((text[position:end], line))
            position = end
    return tokens


def parse_parameter_text(text: str, *, path: str = "<text>") -> ParameterMap:
    """Parse the text of one elastix parameter file into ``{key: (value, ...)}``.

    Raises:
        ValueError: If the text holds anything but entries and comments, an entry is not
            terminated, has no key or a quoted key, or a key repeats; the message names the line.
    """
    entries: ParameterMap = {}
    tokens = _tokens(text, path)
    index = 0
    while index < len(tokens):
        token, line = tokens[index]
        if token != "(":
            raise ValueError(f"{path}:{line}: expected '(' to open an entry, got {token!r}")
        index += 1
        if index >= len(tokens) or tokens[index][0] in "()" or tokens[index][0].startswith('"'):
            raise ValueError(f"{path}:{line}: an entry needs an unquoted key")
        key = tokens[index][0]
        values: list[str] = []
        index += 1
        while index < len(tokens) and tokens[index][0] != ")":
            value, value_line = tokens[index]
            if value == "(":
                raise ValueError(f"{path}:{value_line}: entry {key!r} is not terminated")
            values.append(value[1:] if value.startswith('"') else value)
            index += 1
        if index >= len(tokens):
            raise ValueError(f"{path}:{line}: entry {key!r} is not terminated")
        if key in entries:
            raise ValueError(f"{path}:{line}: key {key!r} repeats")
        entries[key] = tuple(values)
        index += 1
    return entries


def read_parameter_file(path: str | os.PathLike[str]) -> ParameterMap:
    """Read one elastix parameter file, such as ``TransformParameters.0.txt``.

    Args:
        path: The file.

    Returns:
        ``{key: (value, ...)}`` with values as strings and quotes removed, the form itk-elastix
        returns from ``ParameterObject.GetParameterMap``.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If it is not an elastix parameter file; the message names the line.
    """
    file = Path(path)
    return parse_parameter_text(file.read_text(encoding="utf-8"), path=str(file))


def read_parameter_files(path: str | os.PathLike[str]) -> tuple[ParameterMap, ...]:
    """Read the stage chain ending in ``path``, following ``InitialTransformParameterFileName``.

    elastix writes each stage's file naming the previous stage's file, ending at
    ``NoInitialTransform``. A relative name is resolved against the directory of the file that
    names it.

    Args:
        path: The last stage's file, such as ``TransformParameters.1.txt``.

    Returns:
        The stages in application order, the initial transform first.

    Raises:
        FileNotFoundError: If a named initial transform file does not exist; the message names
            the file that refers to it.
        OSError: If a file cannot be read.
        ValueError: If a file is malformed, names more than one initial file, or the chain
            repeats a file.
    """
    stages: list[ParameterMap] = []
    seen: list[Path] = []
    current = Path(path)
    while True:
        resolved = current.resolve()
        if resolved in seen:
            raise ValueError(f"{current} is named twice in the chain of initial transforms")
        seen.append(resolved)
        stage = read_parameter_file(current)
        stages.append(stage)
        initial = stage.get("InitialTransformParameterFileName", (NO_INITIAL_TRANSFORM,))
        if len(initial) != 1:
            raise ValueError(
                f"{current}: InitialTransformParameterFileName must hold one file name, got "
                f"{initial!r}"
            )
        if initial[0] == NO_INITIAL_TRANSFORM:
            break
        named = Path(initial[0])
        following = named if named.is_absolute() else current.parent / named
        if not following.is_file():
            raise FileNotFoundError(
                f"{current} names the initial transform {initial[0]!r}, which is not a file at "
                f"{following}"
            )
        current = following
    return tuple(reversed(stages))
