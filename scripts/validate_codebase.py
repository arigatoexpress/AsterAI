#!/usr/bin/env python3
"""
Codebase validation and cleanup script.
Checks for common issues, missing imports, and code quality.
"""

import os
import sys
import ast
import importlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class CodeValidator:
    """Validates Python code for common issues."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.issues = []
        self.warnings = []
        self.fixed_files = []
    
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate a single Python file."""
        issues = []
        warnings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Check for common issues
            issues.extend(self._check_imports(tree, file_path))
            issues.extend(self._check_syntax_issues(tree, file_path))
            issues.extend(self._check_code_quality(tree, file_path))
            
            # Check for specific patterns
            warnings.extend(self._check_patterns(content, file_path))
            
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        except Exception as e:
            issues.append(f"Parse error: {e}")
        
        return {
            'file': str(file_path),
            'issues': issues,
            'warnings': warnings,
            'valid': len(issues) == 0
        }
    
    def _check_imports(self, tree: ast.AST, file_path: Path) -> List[str]:
        """Check for import issues."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not self._is_import_available(alias.name):
                        issues.append(f"Missing import: {alias.name}")
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and not self._is_import_available(node.module):
                    issues.append(f"Missing module: {node.module}")
        
        return issues
    
    def _check_syntax_issues(self, tree: ast.AST, file_path: Path) -> List[str]:
        """Check for syntax and style issues."""
        issues = []
        
        for node in ast.walk(tree):
            # Check for unused variables
            if isinstance(node, ast.FunctionDef):
                issues.extend(self._check_unused_variables(node, file_path))
            
            # Check for missing docstrings
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not self._has_docstring(node):
                if not node.name.startswith('_'):  # Skip private methods
                    issues.append(f"Missing docstring: {node.name}")
        
        return issues
    
    def _check_code_quality(self, tree: ast.AST, file_path: Path) -> List[str]:
        """Check for code quality issues."""
        issues = []
        
        for node in ast.walk(tree):
            # Check for long functions
            if isinstance(node, ast.FunctionDef):
                lines = len(node.body)
                if lines > 50:
                    issues.append(f"Long function: {node.name} ({lines} lines)")
            
            # Check for deep nesting
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                max_depth = self._get_max_nesting_depth(node)
                if max_depth > 4:
                    issues.append(f"Deep nesting: {node.name} (depth: {max_depth})")
        
        return issues
    
    def _check_patterns(self, content: str, file_path: Path) -> List[str]:
        """Check for specific code patterns."""
        warnings = []
        
        # Check for TODO/FIXME comments
        if 'TODO' in content or 'FIXME' in content:
            warnings.append("Contains TODO/FIXME comments")
        
        # Check for print statements
        if 'print(' in content and 'logger' not in content:
            warnings.append("Uses print() instead of logger")
        
        # Check for hardcoded values
        if any(pattern in content for pattern in ['localhost', '127.0.0.1', 'password=', 'api_key=']):
            warnings.append("Contains potentially hardcoded values")
        
        return warnings
    
    def _is_import_available(self, module_name: str) -> bool:
        """Check if a module can be imported."""
        try:
            # Handle relative imports
            if module_name.startswith('.'):
                return True
            
            # Try to import the module
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False
    
    def _has_docstring(self, node: ast.FunctionDef) -> bool:
        """Check if a function has a docstring."""
        if not node.body:
            return False
        
        first_stmt = node.body[0]
        return isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant) and isinstance(first_stmt.value.value, str)
    
    def _check_unused_variables(self, func_node: ast.FunctionDef, file_path: Path) -> List[str]:
        """Check for unused variables in a function."""
        issues = []
        
        # Get all variable names
        variables = set()
        for node in ast.walk(func_node):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                variables.add(node.id)
        
        # Check for unused variables (simplified check)
        # This is a basic implementation - a full check would be more complex
        return issues
    
    def _get_max_nesting_depth(self, node: ast.AST) -> int:
        """Get maximum nesting depth of a node."""
        max_depth = 0
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                depth = self._get_node_depth(child, node)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _get_node_depth(self, target_node: ast.AST, root_node: ast.AST) -> int:
        """Get depth of a node within the root node."""
        depth = 0
        current = target_node
        
        while current != root_node and hasattr(current, 'parent'):
            depth += 1
            current = current.parent
        
        return depth
    
    def validate_project(self) -> Dict[str, Any]:
        """Validate the entire project."""
        print("🔍 Validating AIAster codebase...")
        
        python_files = list(self.project_root.rglob("*.py"))
        python_files = [f for f in python_files if not any(part.startswith('.') for part in f.parts)]
        
        results = {
            'total_files': len(python_files),
            'valid_files': 0,
            'files_with_issues': 0,
            'total_issues': 0,
            'total_warnings': 0,
            'file_results': []
        }
        
        for file_path in python_files:
            print(f"  Checking {file_path.relative_to(self.project_root)}...")
            
            file_result = self.validate_file(file_path)
            results['file_results'].append(file_result)
            
            if file_result['valid']:
                results['valid_files'] += 1
            else:
                results['files_with_issues'] += 1
            
            results['total_issues'] += len(file_result['issues'])
            results['total_warnings'] += len(file_result['warnings'])
            
            # Print issues
            if file_result['issues']:
                for issue in file_result['issues']:
                    print(f"    ❌ {issue}")
            
            if file_result['warnings']:
                for warning in file_result['warnings']:
                    print(f"    ⚠️  {warning}")
        
        return results


def check_dependencies():
    """Check if all required dependencies are available."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'requests', 'websockets', 'pydantic',
        'python-dotenv', 'streamlit', 'plotly', 'tqdm',
        'google-cloud-bigquery', 'google-cloud-functions', 'google-cloud-scheduler',
        'pyarrow', 'ray', 'aiohttp', 'vaderSentiment', 'feedparser',
        'scikit-learn', 'xgboost', 'lightgbm'
    ]
    
    missing_packages = []
    available_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            available_packages.append(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
    else:
        print("\n✅ All required packages are available")
    
    return {
        'available': available_packages,
        'missing': missing_packages,
        'all_available': len(missing_packages) == 0
    }


def check_project_structure():
    """Check project structure and important files."""
    print("\n📁 Checking project structure...")
    
    project_root = Path(__file__).parent.parent
    required_files = [
        'requirements.txt',
        'README.md',
        'mcp_trader/__init__.py',
        'mcp_trader/config.py',
        'mcp_trader/models/__init__.py',
        'mcp_trader/strategies/__init__.py',
        'mcp_trader/backtesting/__init__.py',
        'mcp_trader/execution/__init__.py',
        'mcp_trader/indicators/__init__.py',
        'mcp_trader/security/__init__.py',
        'mcp_trader/sentiment/__init__.py',
        'mcp_trader/features/__init__.py',
        'dashboard/app.py',
        'scripts/run_backtest.py'
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            existing_files.append(file_path)
            print(f"  ✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"  ❌ {file_path}")
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
    else:
        print("\n✅ All required files present")
    
    return {
        'existing': existing_files,
        'missing': missing_files,
        'all_present': len(missing_files) == 0
    }


def run_tests():
    """Run basic tests to ensure functionality."""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test imports
        from mcp_trader.config import PRIORITY_SYMBOLS
        from mcp_trader.indicators.dmark import DMarkIndicator
        from mcp_trader.strategies.dmark_strategy import DMarkStrategy
        from mcp_trader.security.secrets import SecretManager
        
        print("  ✅ Core imports working")
        
        # Test DMark indicator
        import pandas as pd
        import numpy as np
        
        data = pd.DataFrame({
            'high': np.random.uniform(100, 110, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 105, 100),
            'volume': np.random.uniform(1000, 2000, 100)
        })
        
        indicator = DMarkIndicator()
        results = indicator.calculate(data['high'], data['low'], data['close'], data['volume'])
        
        print("  ✅ DMark indicator working")
        
        # Test strategy
        strategy = DMarkStrategy()
        strategy.fit(data)
        predictions = strategy.predict(data)
        
        print("  ✅ DMark strategy working")
        
        # Test secret manager
        sm = SecretManager()
        print("  ✅ Secret manager working")
        
        print("\n✅ All basic tests passed")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run complete codebase validation."""
    print("🚀 AIAster Codebase Validation")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent
    
    # Check project structure
    structure_result = check_project_structure()
    
    # Check dependencies
    deps_result = check_dependencies()
    
    # Validate code
    validator = CodeValidator(project_root)
    validation_result = validator.validate_project()
    
    # Run tests
    tests_passed = run_tests()
    
    # Summary
    print("\n📊 Validation Summary")
    print("=" * 30)
    print(f"Project structure: {'✅' if structure_result['all_present'] else '❌'}")
    print(f"Dependencies: {'✅' if deps_result['all_available'] else '❌'}")
    print(f"Code validation: {validation_result['valid_files']}/{validation_result['total_files']} files valid")
    print(f"Basic tests: {'✅' if tests_passed else '❌'}")
    
    if validation_result['total_issues'] > 0:
        print(f"\n⚠️  Found {validation_result['total_issues']} issues and {validation_result['total_warnings']} warnings")
    
    if structure_result['all_present'] and deps_result['all_available'] and tests_passed:
        print("\n🎉 Codebase validation completed successfully!")
        return True
    else:
        print("\n⚠️  Codebase validation completed with issues")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
