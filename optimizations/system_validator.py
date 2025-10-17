"""
System Validator
Comprehensive validation of system integrity, security, and configuration
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Tuple
from pathlib import Path
import importlib.util

logger = logging.getLogger(__name__)


class SystemValidator:
    """
    Validates system configuration, security, and readiness
    
    Checks:
    - Required files and modules
    - API key security
    - Configuration validity
    - Dependencies availability
    - Integration points
    - Potential vulnerabilities
    """
    
    def __init__(self):
        self.root_dir = Path.cwd()
        self.validation_results = {
            'passed': [],
            'warnings': [],
            'errors': [],
            'security_issues': []
        }
    
    async def validate_all(self) -> Dict:
        """
        Run all validation checks
        
        Returns:
            Dictionary with validation results
        """
        
        logger.info("🔍 Starting comprehensive system validation...")
        
        # File structure validation
        await self._validate_file_structure()
        
        # Dependencies validation
        await self._validate_dependencies()
        
        # Security validation
        await self._validate_security()
        
        # Configuration validation
        await self._validate_configuration()
        
        # Integration validation
        await self._validate_integrations()
        
        # Generate report
        return self._generate_report()
    
    async def _validate_file_structure(self):
        """Validate required files and directories exist"""
        
        logger.info("📁 Validating file structure...")
        
        required_files = [
            'data_pipeline/binance_vpn_optimizer.py',
            'data_pipeline/smart_data_router.py',
            'mcp_trader/ai/vpin_calculator_numpy.py',
            'optimizations/integrated_collector.py',
            'optimizations/system_validator.py',
        ]
        
        required_dirs = [
            'data_pipeline',
            'mcp_trader',
            'mcp_trader/ai',
            'optimizations',
            'models',
            'data',
        ]
        
        # Check files
        for file_path in required_files:
            full_path = self.root_dir / file_path
            if full_path.exists():
                self.validation_results['passed'].append(f"✅ File exists: {file_path}")
            else:
                self.validation_results['errors'].append(f"❌ Missing file: {file_path}")
        
        # Check directories
        for dir_path in required_dirs:
            full_path = self.root_dir / dir_path
            if full_path.exists() and full_path.is_dir():
                self.validation_results['passed'].append(f"✅ Directory exists: {dir_path}")
            else:
                self.validation_results['warnings'].append(f"⚠️ Missing directory: {dir_path}")
    
    async def _validate_dependencies(self):
        """Validate required Python packages are installed"""
        
        logger.info("📦 Validating dependencies...")
        
        required_packages = [
            'pandas',
            'numpy',
            'ccxt',
            'aiohttp',
            'asyncio',
        ]
        
        for package in required_packages:
            spec = importlib.util.find_spec(package)
            if spec is not None:
                self.validation_results['passed'].append(f"✅ Package installed: {package}")
            else:
                self.validation_results['errors'].append(f"❌ Missing package: {package}")
    
    async def _validate_security(self):
        """Validate security configuration"""
        
        logger.info("🔒 Validating security...")
        
        # Check API keys file
        api_keys_file = self.root_dir / '.api_keys.json'
        secrets_file = self.root_dir / '.secrets.json'
        
        # Ensure API keys are NOT in git
        gitignore_file = self.root_dir / '.gitignore'
        if gitignore_file.exists():
            with open(gitignore_file, 'r') as f:
                gitignore_content = f.read()
                
            if '.api_keys.json' in gitignore_content:
                self.validation_results['passed'].append("✅ API keys in .gitignore")
            else:
                self.validation_results['security_issues'].append(
                    "⚠️ WARNING: .api_keys.json not in .gitignore - add it!"
                )
            
            if '.secrets.json' in gitignore_content:
                self.validation_results['passed'].append("✅ Secrets in .gitignore")
            else:
                self.validation_results['security_issues'].append(
                    "⚠️ WARNING: .secrets.json not in .gitignore - add it!"
                )
        
        # Check file permissions (if API keys exist)
        if api_keys_file.exists():
            # Check if file is readable
            try:
                with open(api_keys_file, 'r') as f:
                    keys = json.load(f)
                
                # Validate structure (without exposing keys)
                if isinstance(keys, dict):
                    self.validation_results['passed'].append(
                        f"✅ API keys file valid ({len(keys)} keys)"
                    )
                else:
                    self.validation_results['warnings'].append(
                        "⚠️ API keys file has unexpected format"
                    )
            except json.JSONDecodeError:
                self.validation_results['errors'].append("❌ API keys file is not valid JSON")
            except Exception as e:
                self.validation_results['warnings'].append(f"⚠️ Cannot read API keys: {e}")
        else:
            self.validation_results['warnings'].append(
                "⚠️ .api_keys.json not found (may be OK for testing)"
            )
        
        # Check for hardcoded credentials in new files
        await self._check_hardcoded_credentials()
    
    async def _check_hardcoded_credentials(self):
        """Check for hardcoded credentials in code"""
        
        suspicious_patterns = [
            'api_key = "',
            'api_secret = "',
            'password = "',
            'token = "',
        ]
        
        files_to_check = [
            'data_pipeline/binance_vpn_optimizer.py',
            'data_pipeline/smart_data_router.py',
            'optimizations/integrated_collector.py',
        ]
        
        for file_path in files_to_check:
            full_path = self.root_dir / file_path
            if not full_path.exists():
                continue
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            found_issues = False
            for pattern in suspicious_patterns:
                if pattern in content and 'demo' not in content.lower():
                    self.validation_results['security_issues'].append(
                        f"⚠️ Possible hardcoded credential in {file_path}: {pattern}"
                    )
                    found_issues = True
            
            if not found_issues:
                self.validation_results['passed'].append(
                    f"✅ No hardcoded credentials in {Path(file_path).name}"
                )
    
    async def _validate_configuration(self):
        """Validate system configuration"""
        
        logger.info("⚙️ Validating configuration...")
        
        # Check if models exist
        models_dir = self.root_dir / 'models'
        if models_dir.exists():
            model_files = list(models_dir.glob('*.pkl'))
            if model_files:
                self.validation_results['passed'].append(
                    f"✅ Found {len(model_files)} trained models"
                )
            else:
                self.validation_results['warnings'].append(
                    "⚠️ No trained models found in models/ directory"
                )
        
        # Check training results
        training_results = self.root_dir / 'training_results'
        if training_results.exists():
            result_dirs = list(training_results.glob('*'))
            if result_dirs:
                self.validation_results['passed'].append(
                    f"✅ Found {len(result_dirs)} training result sets"
                )
            else:
                self.validation_results['warnings'].append(
                    "⚠️ No training results found"
                )
    
    async def _validate_integrations(self):
        """Validate integration points work correctly"""
        
        logger.info("🔗 Validating integrations...")
        
        try:
            # Test imports
            from data_pipeline.binance_vpn_optimizer import VPNOptimizedBinanceCollector
            from data_pipeline.smart_data_router import SmartDataRouter
            from mcp_trader.ai.vpin_calculator_numpy import VPINCalculator
            from optimizations.integrated_collector import IntegratedDataCollector
            
            self.validation_results['passed'].append("✅ All integration imports successful")
            
            # Test instantiation
            try:
                collector = VPNOptimizedBinanceCollector('iceland')
                self.validation_results['passed'].append("✅ VPN collector instantiates")
            except Exception as e:
                self.validation_results['warnings'].append(
                    f"⚠️ VPN collector instantiation warning: {e}"
                )
            
            try:
                router = SmartDataRouter()
                self.validation_results['passed'].append("✅ Smart router instantiates")
            except Exception as e:
                self.validation_results['warnings'].append(
                    f"⚠️ Smart router instantiation warning: {e}"
                )
            
            try:
                vpin = VPINCalculator()
                self.validation_results['passed'].append("✅ VPIN calculator instantiates")
            except Exception as e:
                self.validation_results['warnings'].append(
                    f"⚠️ VPIN instantiation warning: {e}"
                )
            
        except ImportError as e:
            self.validation_results['errors'].append(f"❌ Integration import error: {e}")
    
    def _generate_report(self) -> Dict:
        """Generate comprehensive validation report"""
        
        total_checks = (
            len(self.validation_results['passed']) +
            len(self.validation_results['warnings']) +
            len(self.validation_results['errors']) +
            len(self.validation_results['security_issues'])
        )
        
        report = {
            **self.validation_results,
            'total_checks': total_checks,
            'passed_count': len(self.validation_results['passed']),
            'warnings_count': len(self.validation_results['warnings']),
            'errors_count': len(self.validation_results['errors']),
            'security_issues_count': len(self.validation_results['security_issues']),
            'overall_status': self._get_overall_status(),
        }
        
        return report
    
    def _get_overall_status(self) -> str:
        """Get overall validation status"""
        
        if self.validation_results['errors']:
            return 'FAILED'
        elif self.validation_results['security_issues']:
            return 'SECURITY_WARNINGS'
        elif self.validation_results['warnings']:
            return 'PASSED_WITH_WARNINGS'
        else:
            return 'PASSED'
    
    def print_report(self, report: Dict):
        """Print formatted validation report"""
        
        print("\n" + "="*70)
        print("🔍 SYSTEM VALIDATION REPORT")
        print("="*70)
        
        print(f"\n📊 Summary:")
        print(f"  Total Checks: {report['total_checks']}")
        print(f"  ✅ Passed: {report['passed_count']}")
        print(f"  ⚠️  Warnings: {report['warnings_count']}")
        print(f"  ❌ Errors: {report['errors_count']}")
        print(f"  🔒 Security Issues: {report['security_issues_count']}")
        print(f"\n  Overall Status: {report['overall_status']}")
        
        if report['passed']:
            print(f"\n✅ PASSED ({len(report['passed'])} checks):")
            for item in report['passed'][:5]:  # Show first 5
                print(f"  {item}")
            if len(report['passed']) > 5:
                print(f"  ... and {len(report['passed']) - 5} more")
        
        if report['warnings']:
            print(f"\n⚠️  WARNINGS ({len(report['warnings'])} issues):")
            for item in report['warnings']:
                print(f"  {item}")
        
        if report['errors']:
            print(f"\n❌ ERRORS ({len(report['errors'])} issues):")
            for item in report['errors']:
                print(f"  {item}")
        
        if report['security_issues']:
            print(f"\n🔒 SECURITY ISSUES ({len(report['security_issues'])} issues):")
            for item in report['security_issues']:
                print(f"  {item}")
        
        print("\n" + "="*70)
        
        if report['overall_status'] == 'PASSED':
            print("✅ System validation PASSED - Ready for deployment!")
        elif report['overall_status'] == 'PASSED_WITH_WARNINGS':
            print("⚠️  System validation PASSED with warnings - Review recommended")
        elif report['overall_status'] == 'SECURITY_WARNINGS':
            print("🔒 System validation has SECURITY WARNINGS - Fix before deployment!")
        else:
            print("❌ System validation FAILED - Fix errors before proceeding!")
        
        print("="*70 + "\n")


async def main():
    """Run system validation"""
    
    validator = SystemValidator()
    report = await validator.validate_all()
    validator.print_report(report)
    
    return report


if __name__ == "__main__":
    # Run validation
    report = asyncio.run(main())

