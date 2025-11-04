#!/usr/bin/env python3
"""
Update copyright year ranges in Python source files.

This script searches for copyright notices in Python files and updates
the year range to include the current year.
"""

import os
import re
from datetime import datetime
from pathlib import Path


def update_copyright_in_file(filepath: Path, current_year: int) -> bool:
    """
    Update copyright year in a single file.

    Args:
        filepath: Path to the file to update
        current_year: Current year to update to

    Returns:
        True if file was modified, False otherwise
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Pattern 1: Copyright YYYY-YYYY (range format)
    # Example: # Copyright 2022-2024 -> # Copyright 2022-2025
    pattern1 = re.compile(
        r'(#\s*Copyright\s+)(\d{4})-(\d{4})',
        re.IGNORECASE
    )

    def replace_range(match):
        prefix = match.group(1)
        start_year = match.group(2)
        end_year = match.group(3)

        # Update end year if it's not current
        if int(end_year) < current_year:
            return f"{prefix}{start_year}-{current_year}"
        return match.group(0)

    content = pattern1.sub(replace_range, content)

    # Pattern 2: Copyright YYYY (single year)
    # Example: # Copyright 2024 -> # Copyright 2024-2025 (if not current year)
    pattern2 = re.compile(
        r'(#\s*Copyright\s+)(\d{4})(?!-)',
        re.IGNORECASE
    )

    def replace_single(match):
        prefix = match.group(1)
        year = match.group(2)

        # If it's not the current year, make it a range
        if int(year) < current_year:
            return f"{prefix}{year}-{current_year}"
        return match.group(0)

    content = pattern2.sub(replace_single, content)

    # Only write if content changed
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True

    return False


def main():
    """Main function to update copyright years in all Python files."""
    current_year = datetime.now().year
    repo_root = Path(__file__).parent.parent

    # Directories to exclude
    exclude_dirs = {
        'proto',
        'venv',
        '.venv',
        'env',
        '.env',
        'build',
        'dist',
        '.git',
        '__pycache__',
        '.pytest_cache',
        '.tox',
        'node_modules',
        '.eggs',
        '*.egg-info',
    }

    files_updated = 0
    files_processed = 0

    # Walk through all Python files
    for root, dirs, files in os.walk(repo_root):
        # Remove excluded directories from dirs list (modifies in-place)
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]

        for filename in files:
            if filename.endswith('.py'):
                filepath = Path(root) / filename
                files_processed += 1

                try:
                    if update_copyright_in_file(filepath, current_year):
                        print(f"Updated: {filepath.relative_to(repo_root)}")
                        files_updated += 1
                except Exception as e:
                    print(f"Error processing {filepath.relative_to(repo_root)}: {e}")

    print(f"\nProcessed {files_processed} files, updated {files_updated} files.")


if __name__ == '__main__':
    main()
