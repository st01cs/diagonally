#!/usr/bin/env python3
"""
Scan every skill under CLAUDE_PLUGIN_ROOT, collect their skill-rules.json files,
and merge the content into .claude/skill-rules.json.

Usage:
    python update_skill_rules.py [--plugin-root <path>]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def deep_merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge dictionaries.
    Nested dicts are merged recursively; other values are overwritten by update.
    """
    result = base.copy()

    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value

    return result


def find_skill_rules_files(root_path: Path, exclude_path: Path = None) -> List[Path]:
    """
    Recursively find all skill-rules.json files, excluding the given file.
    Only return files that live alongside a SKILL.md to ensure real skills.
    """
    skill_rules_files = []

    for skill_rules_file in root_path.rglob("skill-rules.json"):
        # Skip the output file itself
        if exclude_path and skill_rules_file.resolve() == exclude_path.resolve():
            continue
        # Only treat the file as valid when SKILL.md exists in the same directory
        if not (skill_rules_file.parent / "SKILL.md").exists():
            continue
        skill_rules_files.append(skill_rules_file)

    return sorted(skill_rules_files)


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load JSON from file; return empty dict on errors or missing file."""
    try:
        if not file_path.exists():
            return {}

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in {file_path}: {e}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"Warning: Error reading {file_path}: {e}", file=sys.stderr)
        return {}


def merge_skill_rules(plugin_root: Path, output_path: Path) -> None:
    """Merge all skill-rules.json files into the output path."""
    # Find every skill-rules.json (excluding the output file itself)
    skill_rules_files = find_skill_rules_files(plugin_root, exclude_path=output_path)

    if not skill_rules_files:
        print(f"No skill-rules.json files found in {plugin_root}")
        return

    print(f"Found {len(skill_rules_files)} skill-rules.json file(s):")
    for file_path in skill_rules_files:
        print(f"  - {file_path}")

    # Start with existing output rules (if any) so we never clobber them
    merged_rules = load_json_file(output_path)
    existing_skill_names = set(merged_rules.keys())

    # Merge the content of every discovered skill-rules.json file
    for skill_rules_file in skill_rules_files:
        rules = load_json_file(skill_rules_file)
        if not rules:
            continue

        filtered_rules = {}
        for skill_name, config in rules.items():
            if skill_name in existing_skill_names:
                print(
                    f"Skipping {skill_name} from {skill_rules_file} "
                    "(already present in output)."
                )
                continue
            filtered_rules[skill_name] = config

        if not filtered_rules:
            continue

        print(f"Merging {skill_rules_file}...")
        merged_rules = deep_merge_dict(merged_rules, filtered_rules)
        existing_skill_names.update(filtered_rules.keys())

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the merged result
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_rules, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully merged skill rules to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Recursively find and merge skill-rules.json files from CLAUDE_PLUGIN_ROOT"
    )
    parser.add_argument(
        "--plugin-root",
        type=str,
        default=None,
        help="Path to CLAUDE_PLUGIN_ROOT directory (default: from environment variable or current directory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for merged skill-rules.json (default: .claude/skill-rules.json relative to plugin root)",
    )

    args = parser.parse_args()

    # Determine plugin root
    if args.plugin_root:
        plugin_root = Path(args.plugin_root).resolve()
    else:
        plugin_root_env = os.environ.get("CLAUDE_PLUGIN_ROOT")
        if plugin_root_env:
            plugin_root = Path(plugin_root_env).resolve()
        else:
            # Default to current working directory
            plugin_root = Path.cwd().resolve()

    if not plugin_root.exists():
        print(
            f"Error: Plugin root directory does not exist: {plugin_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = plugin_root / ".claude" / "skill-rules.json"

    print(f"Plugin root: {plugin_root}")
    print(f"Output path: {output_path}\n")

    merge_skill_rules(plugin_root, output_path)


if __name__ == "__main__":
    main()
