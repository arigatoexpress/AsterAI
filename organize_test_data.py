#!/usr/bin/env python3
"""
Test Data Organization Script

This script organizes and catalogs all test data in the AsterAI project,
creating a structured archive with proper numbering, dating, and documentation.

Features:
- Automatic test data discovery and cataloging
- Date-based organization with proper numbering
- Comprehensive metadata generation
- Cross-reference linking between related data
- Backup and archival capabilities
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
import hashlib
import os
from typing import Dict, List, Tuple, Any

class TestDataOrganizer:
    """Organizes and catalogs all test data in the AsterAI project."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.archive_root = self.project_root / "TEST_DATA_ARCHIVE"
        self.archive_root.mkdir(exist_ok=True)

        # Test data categories and their sources
        self.test_categories = {
            'backtesting': {
                'source_dirs': ['backtest_results'],
                'file_patterns': ['*.json'],
                'description': 'Backtesting results and performance data'
            },
            'training': {
                'source_dirs': ['training_results'],
                'file_patterns': ['*.json', '*.pkl', '*.png', '*.md'],
                'description': 'Model training results and validation data'
            },
            'trading': {
                'source_dirs': ['trading'],
                'file_patterns': ['*.json'],
                'description': 'Live and paper trading results'
            },
            'validation': {
                'source_dirs': ['data'],
                'file_patterns': ['*validation*.json', '*report*.json'],
                'description': 'Data validation and quality reports'
            },
            'testing': {
                'source_dirs': ['tests'],
                'file_patterns': ['*.py'],
                'description': 'Unit test files and test suites'
            },
            'visualization': {
                'source_dirs': ['visual_reports', 'training_results'],
                'file_patterns': ['*.png', '*.html'],
                'description': 'Generated charts and visual reports'
            },
            'logs': {
                'source_dirs': ['.', 'logs'],
                'file_patterns': ['*result*.txt', '*report*.txt', '*test*.txt'],
                'description': 'Log files and text-based reports'
            }
        }

        # Test data registry
        self.test_registry = []
        self.organized_files = []

    def discover_test_data(self) -> Dict[str, List[Path]]:
        """Discover all test data files in the project."""

        print("🔍 Discovering test data files...")
        discovered_data = {}

        for category, config in self.test_categories.items():
            discovered_data[category] = []

            for source_dir in config['source_dirs']:
                source_path = self.project_root / source_dir

                if source_path.exists():
                    # Find all matching files in the source directory
                    for pattern in config['file_patterns']:
                        for file_path in source_path.rglob(pattern):
                            # Skip files in archive directories and cache directories
                            if not any(skip in str(file_path) for skip in ['__pycache__', 'TEST_DATA_ARCHIVE', '.git']):
                                discovered_data[category].append(file_path)

        return discovered_data

    def parse_date_from_filename(self, filename: str) -> str:
        """Extract date information from filename patterns."""

        # Common date patterns in filenames
        date_patterns = [
            r'(\d{8})',  # YYYYMMDD
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{4}\d{2}\d{2})',  # YYYYMMDD
            r'_(\d{8})_',  # _YYYYMMDD_
            r'_(\d{4}\d{2}\d{2})_',  # _YYYYMMDD_
        ]

        for pattern in date_patterns:
            import re
            match = re.search(pattern, filename)
            if match:
                date_str = match.group(1)
                # Convert to standard YYYY-MM-DD format
                if len(date_str) == 8 and date_str.isdigit():
                    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                elif '-' in date_str:
                    return date_str

        return "undated"

    def get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from a test data file."""

        file_stat = file_path.stat()
        filename = file_path.name

        metadata = {
            'original_path': str(file_path),
            'filename': filename,
            'size_bytes': file_stat.st_size,
            'size_mb': round(file_stat.st_size / (1024 * 1024), 2),
            'modified_date': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'extracted_date': self.parse_date_from_filename(filename),
            'file_hash': self.calculate_file_hash(file_path),
            'category': self.categorize_file(file_path),
            'description': self.get_file_description(file_path)
        }

        return metadata

    def categorize_file(self, file_path: Path) -> str:
        """Categorize a file based on its path and content."""

        file_str = str(file_path)

        if 'backtest' in file_str.lower():
            return 'backtesting'
        elif 'training' in file_str.lower():
            return 'training'
        elif 'trading' in file_str.lower():
            return 'trading'
        elif 'validation' in file_str.lower():
            return 'validation'
        elif file_path.suffix in ['.py'] and 'test' in file_str.lower():
            return 'testing'
        elif file_path.suffix in ['.png', '.html']:
            return 'visualization'
        elif file_path.suffix in ['.txt', '.log']:
            return 'logs'
        else:
            return 'other'

    def get_file_description(self, file_path: Path) -> str:
        """Generate a description for a test data file."""

        filename = file_path.name

        if 'backtest' in filename.lower():
            return "Backtesting results and performance metrics"
        elif 'training' in filename.lower():
            return "Model training results and validation data"
        elif 'validation' in filename.lower():
            return "Data validation and quality assessment results"
        elif 'trading' in filename.lower():
            return "Trading system results and performance data"
        elif 'report' in filename.lower():
            return "Generated report or analysis results"
        elif filename.endswith('.pkl'):
            return "Trained machine learning model"
        elif filename.endswith('.png'):
            return "Visualization chart or graph"
        elif filename.endswith('.py') and 'test' in filename.lower():
            return "Unit test file"
        else:
            return "Test data or result file"

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file for integrity verification."""

        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return "hash_error"

    def organize_by_date(self, discovered_data: Dict[str, List[Path]]) -> Dict[str, Dict[str, List[Dict]]]:
        """Organize discovered data by date and create archive structure."""

        print("📅 Organizing test data by date...")

        date_organized = {}

        # Collect all files with their metadata
        all_files = []
        for category, files in discovered_data.items():
            for file_path in files:
                metadata = self.get_file_metadata(file_path)
                all_files.append(metadata)

        # Group by extracted date
        for metadata in all_files:
            date_key = metadata['extracted_date']

            if date_key not in date_organized:
                date_organized[date_key] = {
                    'files': [],
                    'categories': set(),
                    'total_size': 0,
                    'file_count': 0
                }

            date_organized[date_key]['files'].append(metadata)
            date_organized[date_key]['categories'].add(metadata['category'])
            date_organized[date_key]['total_size'] += metadata['size_bytes']
            date_organized[date_key]['file_count'] += 1

        # Sort dates and assign sequential numbers
        sorted_dates = sorted(date_organized.keys(), key=lambda x: x if x != 'undated' else '9999-12-31')

        numbered_dates = {}
        for i, date_key in enumerate(sorted_dates, 1):
            date_info = date_organized[date_key]
            date_info['archive_number'] = i
            date_info['archive_name'] = f"TEST_ARCHIVE_{i:03d}_{date_key}"
            numbered_dates[date_key] = date_info

        return numbered_dates

    def create_archive_structure(self, date_organized: Dict[str, Dict[str, List[Dict]]]) -> None:
        """Create the physical archive structure on disk."""

        print("🏗️  Creating archive directory structure...")

        # Create date-based directories
        for date_key, date_info in date_organized.items():
            if date_key == 'undated':
                continue

            archive_name = date_info['archive_name']
            archive_path = self.archive_root / archive_name

            # Create main archive directory
            archive_path.mkdir(exist_ok=True)

            # Create category subdirectories
            for category in date_info['categories']:
                category_path = archive_path / category
                category_path.mkdir(exist_ok=True)

            # Copy files to appropriate category directories
            for metadata in date_info['files']:
                source_path = Path(metadata['original_path'])
                category = metadata['category']

                # Create unique filename to avoid conflicts
                file_hash = metadata['file_hash'][:8]  # First 8 characters of hash
                new_filename = f"{source_path.stem}_{file_hash}{source_path.suffix}"
                dest_path = archive_path / category / new_filename

                try:
                    shutil.copy2(source_path, dest_path)
                    metadata['archive_path'] = str(dest_path)
                    self.organized_files.append(metadata)
                    print(f"   📁 Copied: {source_path.name} -> {dest_path}")
                except Exception as e:
                    print(f"   ❌ Failed to copy {source_path.name}: {e}")

    def create_metadata_files(self, date_organized: Dict[str, Dict[str, List[Dict]]]) -> None:
        """Create metadata and index files for the archive."""

        print("📋 Creating metadata and index files...")

        # Create main archive index
        archive_index = {
            'archive_created': datetime.now().isoformat(),
            'total_archives': len([d for d in date_organized.keys() if d != 'undated']),
            'total_files': len(self.organized_files),
            'total_size_mb': round(sum(f['size_bytes'] for f in self.organized_files) / (1024 * 1024), 2),
            'archives': {}
        }

        for date_key, date_info in date_organized.items():
            if date_key == 'undated':
                continue

            archive_index['archives'][date_key] = {
                'archive_number': date_info['archive_number'],
                'archive_name': date_info['archive_name'],
                'file_count': date_info['file_count'],
                'total_size_mb': round(date_info['total_size'] / (1024 * 1024), 2),
                'categories': list(date_info['categories']),
                'files': [f['filename'] for f in date_info['files']]
            }

        # Save main index
        with open(self.archive_root / 'ARCHIVE_INDEX.json', 'w') as f:
            json.dump(archive_index, f, indent=2)

        # Create individual archive manifests
        for date_key, date_info in date_organized.items():
            if date_key == 'undated':
                continue

            manifest = {
                'archive_info': {
                    'name': date_info['archive_name'],
                    'number': date_info['archive_number'],
                    'date': date_key,
                    'created': datetime.now().isoformat(),
                    'total_files': date_info['file_count'],
                    'total_size_mb': round(date_info['total_size'] / (1024 * 1024), 2)
                },
                'categories': {},
                'files': []
            }

            for metadata in date_info['files']:
                manifest['files'].append({
                    'filename': metadata['filename'],
                    'category': metadata['category'],
                    'size_mb': metadata['size_mb'],
                    'description': metadata['description'],
                    'archive_path': metadata['archive_path']
                })

                if metadata['category'] not in manifest['categories']:
                    manifest['categories'][metadata['category']] = {
                        'file_count': 0,
                        'total_size_mb': 0,
                        'files': []
                    }

                manifest['categories'][metadata['category']]['file_count'] += 1
                manifest['categories'][metadata['category']]['total_size_mb'] += metadata['size_mb']
                manifest['categories'][metadata['category']]['files'].append(metadata['filename'])

            # Save manifest
            archive_path = self.archive_root / date_info['archive_name']
            with open(archive_path / 'MANIFEST.json', 'w') as f:
                json.dump(manifest, f, indent=2)

    def create_summary_report(self) -> None:
        """Create a comprehensive summary report of the organized test data."""

        print("📊 Creating summary report...")

        summary = {
            'organization_summary': {
                'total_archives_created': len([f for f in self.organized_files if 'archive_path' in f]),
                'total_files_organized': len(self.organized_files),
                'total_size_mb': round(sum(f['size_bytes'] for f in self.organized_files) / (1024 * 1024), 2),
                'organization_date': datetime.now().isoformat(),
                'categories_found': {}
            },
            'file_details': []
        }

        # Categorize files by type
        categories = {}
        for file_info in self.organized_files:
            category = file_info['category']
            if category not in categories:
                categories[category] = {
                    'count': 0,
                    'total_size_mb': 0,
                    'files': []
                }

            categories[category]['count'] += 1
            categories[category]['total_size_mb'] += file_info['size_mb']
            categories[category]['files'].append(file_info['filename'])

        summary['organization_summary']['categories_found'] = categories

        # Add file details
        for file_info in self.organized_files:
            summary['file_details'].append({
                'filename': file_info['filename'],
                'category': file_info['category'],
                'size_mb': file_info['size_mb'],
                'extracted_date': file_info['extracted_date'],
                'description': file_info['description']
            })

        # Save summary
        with open(self.archive_root / 'ORGANIZATION_SUMMARY.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Create human-readable summary
        with open(self.archive_root / 'README.md', 'w', encoding='utf-8') as f:
            f.write("# 🗂️ Test Data Archive - Organization Summary\n\n")
            f.write("## 📊 Archive Overview\n\n")
            f.write(".1f")
            f.write(f"- **Total Files Organized**: {len(self.organized_files)}\n")
            f.write(".2f")
            f.write(f"- **Organization Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 📁 Categories Organized\n\n")
            for category, info in categories.items():
                f.write(f"### {category.title()}\n")
                f.write(".1f")
                f.write(".2f")
                f.write(f"- **Files**: {', '.join(info['files'][:5])}{'...' if len(info['files']) > 5 else ''}\n\n")

            f.write("## 📅 Archive Structure\n\n")
            f.write("Archives are organized by date with the following naming convention:\n")
            f.write("`TEST_ARCHIVE_XXX_YYYY-MM-DD/` where XXX is the sequential number\n\n")

            f.write("## 🔍 Finding Files\n\n")
            f.write("Each archive contains:\n")
            f.write("- `MANIFEST.json` - Detailed file listing and metadata\n")
            f.write("- Category subdirectories with organized files\n")
            f.write("- Backup of original file structure and metadata\n\n")

            f.write("## 📋 Main Archive Index\n\n")
            f.write("See `ARCHIVE_INDEX.json` for the complete catalog of all organized test data.\n")

    def run_organization(self) -> None:
        """Run the complete test data organization process."""

        print("="*80)
        print("🗂️  ASTER AI TEST DATA ORGANIZATION")
        print("="*80)

        # Discover test data
        discovered_data = self.discover_test_data()

        # Display what was found
        total_files = sum(len(files) for files in discovered_data.values())
        print(".0f")

        for category, files in discovered_data.items():
            print(f"   • {category.title()}: {len(files)} files")

        print()

        # Organize by date
        date_organized = self.organize_by_date(discovered_data)

        # Create archive structure
        self.create_archive_structure(date_organized)

        # Create metadata files
        self.create_metadata_files(date_organized)

        # Create summary report
        self.create_summary_report()

        print("\n" + "="*80)
        print("✅ TEST DATA ORGANIZATION COMPLETE!")
        print("="*80)

        print("📂 Archive Location: TEST_DATA_ARCHIVE/")
        print("📋 Main Index: ARCHIVE_INDEX.json")
        print("📊 Summary: ORGANIZATION_SUMMARY.json")
        print("📖 Human-readable summary: README.md")

        print(".0f")
        print(".2f")

        print("\n🎯 Archive Structure:")
        for date_key, date_info in sorted(date_organized.items(), key=lambda x: x[1]['archive_number']):
            if date_key != 'undated':
                print(f"   {date_info['archive_number']:03d'}")
        print("\n" + "="*80)

def main():
    """Main function to organize test data."""

    # Get project root (assuming script is run from project root)
    project_root = Path.cwd()

    # Create organizer and run organization
    organizer = TestDataOrganizer(project_root)
    organizer.run_organization()

if __name__ == "__main__":
    main()
