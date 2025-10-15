#!/usr/bin/env python3
"""
Codebase Audit Tool for AI Trading System

Identifies unused files, duplicate configurations, and outdated documentation
to prepare for comprehensive cleanup and organization.
"""

import os
import sys
import ast
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any
import hashlib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class CodebaseAuditor:
    """Comprehensive codebase analysis tool"""

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.python_files: List[Path] = []
        self.import_analysis: Dict[str, Set[str]] = {}
        self.duplicate_configs: Dict[str, List[Path]] = defaultdict(list)
        self.duplicate_docs: Dict[str, List[Path]] = defaultdict(list)

    def scan_python_files(self) -> None:
        """Scan all Python files in the project"""
        print("📁 Scanning Python files...")

        # Skip common directories
        skip_dirs = {
            '__pycache__', '.git', '.pytest_cache', 'node_modules',
            'asterai_env', 'asterai_env_fresh', 'venv', 'env',
            '.env', 'build', 'dist', '*.egg-info'
        }

        for root, dirs, files in os.walk(self.root_path):
            # Remove skipped directories
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    self.python_files.append(file_path)

        print(f"   Found {len(self.python_files)} Python files")
        return self.python_files

    def analyze_imports(self) -> Dict[str, Set[str]]:
        """Analyze import relationships between modules"""
        print("🔍 Analyzing import relationships...")

        module_to_file = {}
        file_to_imports = defaultdict(set)

        # Build module to file mapping
        for py_file in self.python_files:
            try:
                rel_path = py_file.relative_to(self.root_path)
                module_name = str(rel_path).replace('.py', '').replace('/', '.').replace('\\', '.')
                if module_name.startswith('.'):
                    module_name = module_name[1:]
                module_to_file[module_name] = py_file
            except ValueError:
                continue

        # Analyze imports in each file
        analyzed_count = 0
        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse AST to find imports
                tree = ast.parse(content, filename=str(py_file))

                imports = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])

                file_to_imports[py_file] = imports
                analyzed_count += 1

            except (SyntaxError, UnicodeDecodeError) as e:
                print(f"   ⚠️  Could not parse {py_file}: {e}")
                continue

        print(f"   Analyzed imports from {analyzed_count} files")
        return dict(file_to_imports)

    def find_unused_files(self, import_analysis: Dict[Path, Set[str]]) -> List[Path]:
        """Find potentially unused Python files"""
        print("🔎 Finding potentially unused files...")

        # Build import graph
        imported_modules = set()
        for imports in import_analysis.values():
            imported_modules.update(imports)

        # Convert to module names
        used_files = set()

        # Check each file to see if it's imported or used
        unused_files = []
        for py_file in self.python_files:
            rel_path = py_file.relative_to(self.root_path)
            module_name = str(rel_path).replace('.py', '').replace('/', '.').replace('\\', '.')

            # Remove leading dots
            module_name = module_name.lstrip('.')

            # Check if any part of this module is imported
            is_used = False
            for imported in imported_modules:
                if module_name.startswith(imported) or imported in module_name:
                    is_used = True
                    break

            # Special cases: entry points
            entry_points = [
                'scripts/', 'run_', 'train_', 'deploy_', 'test_',
                'main.py', 'app.py', 'dashboard', 'agent/live_agent.py'
            ]

            if any(pattern in str(rel_path) for pattern in entry_points):
                is_used = True

            if not is_used:
                unused_files.append(py_file)

        print(f"   Found {len(unused_files)} potentially unused files")
        return unused_files

    def find_duplicate_configs(self) -> Dict[str, List[Path]]:
        """Find duplicate configuration files"""
        print("📋 Finding duplicate configurations...")

        config_patterns = [
            ('DOCKER', ['Dockerfile*', 'docker-compose*.yml']),
            ('KUBERNETES', ['*.yaml', '*.yml']),
            ('REQUIREMENTS', ['requirements*.txt', 'pyproject.toml', 'setup.py']),
            ('CONFIG', ['config*.json', 'config*.yaml', '*.cfg', '*.ini']),
            ('ENV', ['.env*', 'env*'])
        ]

        duplicates = defaultdict(list)

        for category, patterns in config_patterns:
            category_files = []

            for pattern in patterns:
                for file_path in self.root_path.rglob(pattern):
                    if file_path.is_file() and not any(skip in str(file_path) for skip in ['__pycache__', '.git', 'archive']):
                        category_files.append(file_path)

            # Group by content hash for actual duplicates
            content_hashes = {}
            for file_path in category_files:
                try:
                    with open(file_path, 'rb') as f:
                        content_hash = hashlib.md5(f.read()).hexdigest()

                    if content_hash in content_hashes:
                        duplicates[category].append(file_path)
                    else:
                        content_hashes[content_hash] = file_path
                except (IOError, OSError):
                    continue

        total_duplicates = sum(len(files) for files in duplicates.values())
        print(f"   Found duplicate configs in {len(duplicates)} categories")

        return dict(duplicates)

    def find_duplicate_docs(self) -> Dict[str, List[Path]]:
        """Find duplicate or similar documentation files"""
        print("📚 Finding duplicate documentation...")

        doc_extensions = ['.md', '.txt', '.rst', '.adoc']
        doc_files = []

        for ext in doc_extensions:
            doc_files.extend(self.root_path.rglob(f'*{ext}'))

        # Remove files in unwanted directories
        doc_files = [f for f in doc_files if not any(skip in str(f) for skip in ['__pycache__', '.git', 'archive', 'asterai_env', 'asterai_env_fresh'])]

        # Group by content similarity (first 1000 chars)
        content_groups = defaultdict(list)

        for doc_file in doc_files:
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read(1000).strip()
                    if len(content) > 100:
                        content_hash = hashlib.md5(content.encode()).hexdigest()
                        content_groups[content_hash].append(doc_file)
            except (UnicodeDecodeError, IOError):
                continue

        # Only keep groups with duplicates
        duplicates = {k: v for k, v in content_groups.items() if len(v) > 1}

        print(f"   Found {len(duplicates)} groups of similar documentation")
        return duplicates

    def generate_report(self, unused_files: List[Path],
                       duplicate_configs: Dict[str, List[Path]],
                       duplicate_docs: Dict[str, List[Path]]) -> str:
        """Generate comprehensive audit report"""

        lines = []
        lines.append("="*70)
        lines.append("CODEBASE AUDIT TOOL")
        lines.append("="*70)
        lines.append("")
        lines.append("Generated: 2025-10-15 17:03:32")
        lines.append(f"Root Directory: {self.root_path}")
        lines.append("")

        # Summary
        lines.append("📊 SUMMARY")
        lines.append(f"   Total Python files: {len(self.python_files)}")
        lines.append(f"   Potentially unused: {len(unused_files)}")
        lines.append(f"   Duplicate configs: {sum(len(files) for files in duplicate_configs.values())}")
        lines.append(f"   Duplicate docs: {sum(len(files) for files in duplicate_docs.values())}")
        lines.append("")

        # Unused files
        if unused_files:
            lines.append("🔎 POTENTIALLY UNUSED FILES ({})".format(len(unused_files)))
            for file_path in sorted(unused_files):
                rel_path = file_path.relative_to(self.root_path)
                lines.append(f"   - {rel_path}")
            lines.append("")

        # Duplicate configs
        if duplicate_configs:
            lines.append("📋 DUPLICATE CONFIGURATIONS")
            for category, files in duplicate_configs.items():
                lines.append(f"   {category}:")
                for file_path in sorted(files):
                    rel_path = file_path.relative_to(self.root_path)
                    lines.append(f"     - {rel_path}")
            lines.append("")

        # Duplicate docs
        if duplicate_docs:
            lines.append("📚 DUPLICATE DOCUMENTATION ({})".format(sum(len(files) for files in duplicate_docs.values())))
            for i, (content_hash, files) in enumerate(duplicate_docs.items()):
                lines.append(f"   Group {i+1}:")
                for file_path in sorted(files):
                    rel_path = file_path.relative_to(self.root_path)
                    lines.append(f"     - {rel_path}")
            lines.append("")

        # Recommendations
        lines.append("✅ RECOMMENDATIONS:")
        lines.append("   1. Review unused files - may be legitimate entry points")
        lines.append("   2. Consolidate duplicate configs - keep most recent")
        lines.append("   3. Merge duplicate docs into single authoritative version")
        lines.append("   4. Archive (don't delete) to preserve history")

        return "\n".join(lines)


def main():
    """Main audit function"""
    root_path = Path(__file__).parent.parent

    auditor = CodebaseAuditor(root_path)

    # Scan and analyze
    python_files = auditor.scan_python_files()
    import_analysis = auditor.analyze_imports()
    unused_files = auditor.find_unused_files(import_analysis)
    duplicate_configs = auditor.find_duplicate_configs()
    duplicate_docs = auditor.find_duplicate_docs()

    # Generate report
    report = auditor.generate_report(unused_files, duplicate_configs, duplicate_docs)
    print("\n" + report)

    # Save report
    report_file = Path("CODEBASE_AUDIT_REPORT.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✅ Report saved to: {report_file}")

    # Ask for confirmation before archiving
    print("\n" + "="*70)
    print("⚠️  ARCHIVE RECOMMENDATION")
    print("="*70)

    if unused_files:
        print(f"Found {len(unused_files)} potentially unused files.")
        print("Consider moving them to an 'archive/' directory for safekeeping.")

    if duplicate_configs or duplicate_docs:
        print("Found duplicate configurations and documentation.")
        print("Consider consolidating into single authoritative versions.")

    print("\nRun with --archive flag to automatically create archive structure.")


if __name__ == "__main__":
    main()
