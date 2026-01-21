#!/usr/bin/env python3
"""Edit TOML config files while preserving comments and formatting."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""

    RED = "\033[91m"      # Deletions
    GREEN = "\033[92m"    # Additions
    YELLOW = "\033[93m"   # Replacements
    BLUE = "\033[94m"     # Info
    RESET = "\033[0m"     # Reset
    BOLD = "\033[1m"


def parse_value(value_str: str) -> str:
    """Format value based on type inference."""
    # Check if it's a TOML array format
    # Matches: [1, 2, 3], [1, 2.5, 3], [true, false], [1, 2, 3, 4], etc.
    array_pattern = re.compile(r'^\s*\[.*\]\s*$')
    if array_pattern.match(value_str):
        return value_str  # Return array as-is without quotes

    # Boolean
    if value_str.lower() in ("true", "false"):
        return value_str.lower()

    # Try int
    try:
        int(value_str)
        return value_str
    except ValueError:
        pass

    # Try float
    try:
        float(value_str)
        return value_str
    except ValueError:
        pass

    # String - add quotes if not already quoted
    if value_str.startswith('"') and value_str.endswith('"'):
        return value_str
    return f'"{value_str}"'


def parse_setting(setting: str) -> tuple[list[str], str, str]:
    """
    Parse a setting string like 'logging.wandb_project=signal-diffusion'.

    Returns:
        (section_parts, key, formatted_value)
        e.g., (['logging'], 'wandb_project', '"signal-diffusion"')
    """
    if "=" not in setting:
        raise ValueError(f"Setting must contain '=': {setting}")

    path, value = setting.split("=", 1)
    parts = path.split(".")

    if len(parts) < 2:
        raise ValueError(f"Setting must have at least one section: {setting}")

    section_parts = parts[:-1]
    key = parts[-1]
    formatted_value = parse_value(value)

    return section_parts, key, formatted_value


def parse_delete_path(path: str) -> tuple[list[str], str]:
    """
    Parse a delete path like 'logging.wandb_project'.

    Returns:
        (section_parts, key)
        e.g., (['logging'], 'wandb_project')
    """
    parts = path.split(".")

    if len(parts) < 2:
        raise ValueError(f"Delete path must have at least one section: {path}")

    section_parts = parts[:-1]
    key = parts[-1]

    return section_parts, key


def find_section_range(lines: list[str], section_parts: list[str]) -> tuple[int, int]:
    """
    Find the start and end line indices for a section.

    Returns:
        (start_idx, end_idx) where start_idx is the line with [section],
        and end_idx is the last line before the next section (or EOF).
        Returns (-1, -1) if section not found.
    """
    section_header = "[" + ".".join(section_parts) + "]"
    section_pattern = re.compile(r"^\[([^\]]+)\]")

    start_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == section_header:
            start_idx = i
            break

    if start_idx == -1:
        return -1, -1

    # Find end of section (next section or EOF)
    end_idx = len(lines) - 1
    for i in range(start_idx + 1, len(lines)):
        if section_pattern.match(lines[i].strip()):
            end_idx = i - 1
            break

    return start_idx, end_idx


def find_last_setting_in_section(lines: list[str], start_idx: int, end_idx: int) -> int:
    """Find the last line with a setting (key = value) in the section."""
    setting_pattern = re.compile(r"^\s*\w+\s*=")

    last_setting_idx = start_idx
    for i in range(start_idx + 1, end_idx + 1):
        line = lines[i]
        if setting_pattern.match(line):
            last_setting_idx = i

    return last_setting_idx


def delete_setting(
    lines: list[str],
    section_parts: list[str],
    key: str,
) -> tuple[list[str], bool]:
    """
    Delete a setting from the section.

    Returns:
        (modified_lines, found) where found indicates if the setting existed
    """
    start_idx, end_idx = find_section_range(lines, section_parts)

    if start_idx == -1:
        # Section doesn't exist
        return lines, False

    # Look for existing key in section
    key_pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")

    for i in range(start_idx + 1, end_idx + 1):
        if key_pattern.match(lines[i]):
            # Found the setting - delete it
            lines.pop(i)
            return lines, True

    return lines, False


def update_or_insert_setting(
    lines: list[str],
    section_parts: list[str],
    key: str,
    value: str,
) -> tuple[list[str], str, str | None]:
    """
    Update existing setting or insert new one in the section.

    Returns:
        (modified_lines, operation_type, old_value)
        where operation_type is "added" or "updated"
        and old_value is the previous value (or None for additions)
    """
    start_idx, end_idx = find_section_range(lines, section_parts)

    if start_idx == -1:
        # Section doesn't exist - create it at the end
        if lines and not lines[-1].strip():
            # Remove trailing blank lines
            while lines and not lines[-1].strip():
                lines.pop()

        lines.append("\n")
        lines.append(f"[{'.'.join(section_parts)}]\n")
        lines.append(f"{key} = {value}\n")
        return lines, "added", None

    # Look for existing key in section
    key_pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")

    for i in range(start_idx + 1, end_idx + 1):
        if key_pattern.match(lines[i]):
            # Capture old value before updating
            old_line = lines[i].strip()
            old_value = old_line.split("=", 1)[1].strip() if "=" in old_line else "?"

            # Update existing setting
            indent = len(lines[i]) - len(lines[i].lstrip())
            lines[i] = f"{' ' * indent}{key} = {value}\n"
            return lines, "updated", old_value

    # Key doesn't exist - insert after last setting in section
    last_setting_idx = find_last_setting_in_section(lines, start_idx, end_idx)

    # Determine indentation from existing settings
    indent = 0
    if last_setting_idx > start_idx:
        indent = len(lines[last_setting_idx]) - len(lines[last_setting_idx].lstrip())

    # Insert new setting
    new_line = f"{' ' * indent}{key} = {value}\n"
    lines.insert(last_setting_idx + 1, new_line)

    return lines, "added", None


def edit_config(
    config_path: Path,
    settings: list[str] | None = None,
    deletes: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """Edit a config file with the given settings and deletions."""
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Read file
    with config_path.open("r") as f:
        lines = f.readlines()

    modified = False

    # Apply deletions first
    if deletes:
        for delete_path in deletes:
            section_parts, key = parse_delete_path(delete_path)
            lines, found = delete_setting(lines, section_parts, key)
            if found:
                path = f"{'.'.join(section_parts)}.{key}"
                print(f"  {Colors.RED}✗{Colors.RESET} {Colors.BOLD}{path}{Colors.RESET}")
                modified = True
            else:
                path = f"{'.'.join(section_parts)}.{key}"
                print(f"  {Colors.BLUE}ℹ{Colors.RESET} {path} (not found)")

    # Apply settings
    if settings:
        for setting in settings:
            section_parts, key, value = parse_setting(setting)
            lines, operation, old_value = update_or_insert_setting(lines, section_parts, key, value)
            path = f"{'.'.join(section_parts)}.{key}"

            if operation == "added":
                print(f"  {Colors.GREEN}+{Colors.RESET} {Colors.BOLD}{path}{Colors.RESET} = {value}")
            else:  # updated
                print(f"  {Colors.YELLOW}~{Colors.RESET} {Colors.BOLD}{path}{Colors.RESET}: {old_value} → {value}")

            modified = True

    # Write back
    if modified:
        if dry_run:
            print(f"\n{Colors.BLUE}Dry run - would write to {config_path}{Colors.RESET}")
        else:
            with config_path.open("w") as f:
                f.writelines(lines)
            print(f"\n{Colors.GREEN}✓ Updated {config_path}{Colors.RESET}")
    else:
        print(f"\nNo changes made to {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Edit TOML config files while preserving comments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enable wandb logging
  %(prog)s -c config/diffusion/baseline.toml \\
    -s logging.wandb_project=signal-diffusion

  # Update multiple settings (multiple -s flags or space-separated)
  %(prog)s -c config/diffusion/baseline.toml \\
    -s logging.wandb_project=signal-diffusion logging.wandb_entity=myteam \\
    -s model.extras.in_channels=3

  # Delete settings
  %(prog)s -c config/diffusion/baseline.toml \\
    -d logging.wandb_entity logging.wandb_project

  # Update multiple config files with glob
  %(prog)s -c config/diffusion/*.toml \\
    -s logging.wandb_project=signal-diffusion

  # Delete and add in one command
  %(prog)s -c config/diffusion/baseline.toml \\
    -d logging.tensorboard \\
    -s logging.wandb_project=signal-diffusion

  # Mix multiple files and flags
  %(prog)s -c config1.toml config2.toml -c config3.toml \\
    -s training.epochs=50
        """,
    )

    parser.add_argument(
        "-c",
        "--config",
        action="extend",
        nargs="+",
        required=True,
        metavar="FILE",
        help="Config file(s) to edit (accepts multiple files and globs)",
    )
    parser.add_argument(
        "-s",
        "--setting",
        action="extend",
        nargs="+",
        metavar="KEY=VALUE",
        help="Setting(s) to modify in format 'section.key=value'",
    )
    parser.add_argument(
        "-d",
        "--delete",
        action="extend",
        nargs="+",
        metavar="KEY",
        help="Setting(s) to delete in format 'section.key'",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )

    args = parser.parse_args()

    # Validate that at least one of -s or -d is provided
    if not args.setting and not args.delete:
        parser.error("At least one of -s/--setting or -d/--delete must be provided")

    # Validate no overlap between settings and deletes
    if args.setting and args.delete:
        setting_paths = set()
        for setting in args.setting:
            path = setting.split("=", 1)[0]
            setting_paths.add(path)

        delete_paths = set(args.delete)
        conflicts = setting_paths & delete_paths

        if conflicts:
            parser.error(
                f"Cannot both modify and delete the same settings: {', '.join(sorted(conflicts))}"
            )

    # Process each config file
    for config_str in args.config:
        config_path = Path(config_str).expanduser().resolve()
        print(f"\nEditing {config_path}:")
        edit_config(config_path, args.setting, args.delete, args.dry_run)


if __name__ == "__main__":
    main()
