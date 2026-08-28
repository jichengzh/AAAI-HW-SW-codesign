"""Exact non-constructing tag policy for trusted model-native YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode


_ALLOWED_TAGS = frozenset(
    {
        "tag:yaml.org,2002:binary",
        "tag:yaml.org,2002:map",
        "tag:yaml.org,2002:python/name:numpy.ndarray",
        "tag:yaml.org,2002:python/object/apply:collections.OrderedDict",
        "tag:yaml.org,2002:python/object/apply:numpy.core.multiarray._reconstruct",
        "tag:yaml.org,2002:python/object/apply:numpy.dtype",
        "tag:yaml.org,2002:python/tuple",
        "tag:yaml.org,2002:seq",
        "tag:yaml.org,2002:str",
    }
)


def validate_model_native_yaml_tags(path: Path) -> None:
    """Reject unobserved tags without invoking any YAML constructor."""

    try:
        documents = tuple(yaml.compose_all(path.read_bytes(), Loader=yaml.BaseLoader))
        if len(documents) != 1 or documents[0] is None:
            raise ValueError
        if not _node_tags(documents[0]).issubset(_ALLOWED_TAGS):
            raise ValueError
    except (OSError, TypeError, UnicodeError, ValueError, yaml.YAMLError) as error:
        raise ValueError("formal model config tags are invalid") from error


def _node_tags(root: Node) -> frozenset[str]:
    tags: set[str] = set()
    pending = [root]
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        tags.add(node.tag)
        if isinstance(node, MappingNode):
            pending.extend(child for pair in node.value for child in pair)
        elif isinstance(node, SequenceNode):
            pending.extend(node.value)
        elif not isinstance(node, ScalarNode):
            raise ValueError("formal model config node is invalid")
    return frozenset(tags)


__all__ = ["validate_model_native_yaml_tags"]
