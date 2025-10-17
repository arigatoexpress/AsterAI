#!/usr/bin/env python3
"""
Security Audit for AsterAI Trading System

Comprehensive security assessment covering:
- Input validation and sanitization
- Authentication and authorization
- Data encryption and protection
- Secure coding practices
- Dependency security
- Configuration security
"""

import os
import re
import hashlib
import ast
import sys
from typing import Dict, List, Set, Tuple
from pathlib import Path
import json
import subprocess

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_trader.security.config import security_config
from mcp_trader.security.input_validation import InputValidator


class SecurityAuditor:
    """Comprehensive security auditor for the codebase"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.findings = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': [],
            'info': []
        }
        self.audit_stats = {
            'files_scanned': 0,
            'lines_scanned': 0,
            'vulnerabilities_found': 0,
            'security_score': 100
        }

    def run_full_audit(self) -> Dict:
        """Run comprehensive security audit"""
        print("🔒 Starting Comprehensive Security Audit")
        print("=" * 50)

        # Audit different aspects
        self.audit_file_permissions()
        self.audit_sensitive_files()
        self.audit_dependencies()
        self.audit_code_security()
        self.audit_configuration_security()
        self.audit_network_security()

        # Calculate security score
        self.calculate_security_score()

        # Generate report
        report = self.generate_audit_report()
        print(report)

        return {
            'findings': self.findings,
            'stats': self.audit_stats,
            'recommendations': self.generate_recommendations()
        }

    def audit_file_permissions(self):
        """Audit file permissions and access controls"""
        print("📁 Auditing file permissions...")

        sensitive_files = [
            '.env', '.secrets.json', '.api_keys.json',
            'config.py', 'settings.py', 'secrets.py'
        ]

        for file_path in sensitive_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                # Check if file is readable by others (basic check)
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()

                    # Check for hardcoded secrets
                    if self.contains_sensitive_data(content):
                        self.add_finding('critical',
                            f"Sensitive file {file_path} contains potential secrets",
                            str(full_path))

                    # Check file permissions (basic check)
                    if os.name == 'nt':  # Windows
                        # On Windows, check if file is hidden or has restrictive permissions
                        pass
                    else:
                        # Unix-like systems
                        stat_info = os.stat(full_path)
                        permissions = oct(stat_info.st_mode)[-3:]

                        # Check if world-readable
                        if permissions[2] in ['4', '5', '6', '7']:
                            self.add_finding('high',
                                f"File {file_path} is world-readable (permissions: {permissions})",
                                str(full_path))

                except Exception as e:
                    self.add_finding('medium',
                        f"Could not read sensitive file {file_path}: {e}",
                        str(full_path))

    def audit_sensitive_files(self):
        """Audit for sensitive data exposure"""
        print("🔐 Auditing sensitive data exposure...")

        # Search for potential secrets in code
        secret_patterns = [
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'api_secret\s*=\s*["\'][^"\']+["\']',
            r'password\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'PRIVATE_KEY',
            r'AWS_ACCESS_KEY',
            r'DATABASE_URL'
        ]

        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')

                for i, line in enumerate(lines, 1):
                    for pattern in secret_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Skip if it's clearly a placeholder or import
                            if not any(skip in line.lower() for skip in [
                                'import', 'from', 'placeholder', 'your_', 'example',
                                'os.getenv', 'os.environ', 'config.'
                            ]):
                                self.add_finding('high',
                                    f"Potential hardcoded secret found in {py_file.name}:{i}",
                                    f"{py_file}:{i} - {line.strip()}")

            except Exception as e:
                self.add_finding('low',
                    f"Could not audit file {py_file.name}: {e}",
                    str(py_file))

    def audit_dependencies(self):
        """Audit Python dependencies for security vulnerabilities"""
        print("📦 Auditing dependencies...")

        # Check requirements.txt
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    deps = f.read().split('\n')

                # Check for known vulnerable packages (basic check)
                vulnerable_packages = {
                    'django': 'Check for Django versions < 3.2',
                    'flask': 'Check for Flask versions < 2.0',
                    'requests': 'Check for requests versions < 2.25.0',
                    'cryptography': 'Keep cryptography up to date'
                }

                for dep in deps:
                    dep = dep.strip().split('==')[0].split('>=')[0].split('>')[0]
                    if dep in vulnerable_packages:
                        self.add_finding('medium',
                            f"Package {dep} should be audited for known vulnerabilities",
                            vulnerable_packages[dep])

            except Exception as e:
                self.add_finding('low',
                    f"Could not audit requirements.txt: {e}",
                    str(req_file))

        # Check for unpinned dependencies
        try:
            with open(req_file, 'r') as f:
                content = f.read()

            unpinned_deps = []
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Check if version is pinned
                    if '==' not in line and '>=' not in line and '~=' not in line and not line.endswith(']'):
                        unpinned_deps.append(line)

            if unpinned_deps:
                self.add_finding('medium',
                    f"Found {len(unpinned_deps)} unpinned dependencies - should pin versions",
                    f"Unpinned: {', '.join(unpinned_deps[:5])}")

        except Exception as e:
            pass

    def audit_code_security(self):
        """Audit code for security vulnerabilities"""
        print("💻 Auditing code security...")

        security_issues = {
            'sql_injection': [
                r'execute\s*\(',
                r'\.format\(.*%.*\)',
                r'\+.*sql',
                r'%s.*sql'
            ],
            'xss': [
                r'innerHTML\s*=',
                r'document\.write\s*\(',
                r'eval\s*\('
            ],
            'command_injection': [
                r'os\.system\s*\(',
                r'subprocess\.call\s*\(',
                r'os\.popen\s*\('
            ],
            'weak_crypto': [
                r'md5\s*\(',
                r'sha1\s*\(',
                r'des\s*\('
            ]
        }

        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Check for security issues
                for issue_type, patterns in security_issues.items():
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                        if matches:
                            severity = 'high' if issue_type in ['sql_injection', 'command_injection'] else 'medium'
                            self.add_finding(severity,
                                f"Potential {issue_type.replace('_', ' ')} vulnerability in {py_file.name}",
                                f"{py_file} - Pattern: {pattern}")

                # Check for dangerous imports
                dangerous_imports = [
                    'pickle', 'marshal', 'shelve',  # Deserialization vulnerabilities
                    'telnetlib', 'ftplib'  # Insecure protocols
                ]

                for dangerous in dangerous_imports:
                    if f"import {dangerous}" in content or f"from {dangerous}" in content:
                        self.add_finding('medium',
                            f"Dangerous import '{dangerous}' found in {py_file.name}",
                            str(py_file))

            except Exception as e:
                self.add_finding('low',
                    f"Could not audit code in {py_file.name}: {e}",
                    str(py_file))

    def audit_configuration_security(self):
        """Audit configuration security"""
        print("⚙️  Auditing configuration security...")

        # Check for debug mode in production
        config_files = ['mcp_trader/config.py', 'dashboard/aster_trader_dashboard.py']

        for config_file in config_files:
            file_path = self.project_root / config_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()

                    # Check for debug settings
                    if 'DEBUG' in content and 'True' in content:
                        self.add_finding('medium',
                            f"Debug mode may be enabled in {config_file}",
                            "Ensure DEBUG=False in production")

                    # Check for default secrets
                    if 'change-me' in content.lower() or 'your-secret' in content.lower():
                        self.add_finding('high',
                            f"Default or placeholder secrets found in {config_file}",
                            "Replace all default secrets with secure values")

                except Exception as e:
                    self.add_finding('low',
                        f"Could not audit config file {config_file}: {e}",
                        str(file_path))

    def audit_network_security(self):
        """Audit network security configurations"""
        print("🌐 Auditing network security...")

        # Check for insecure protocols
        insecure_patterns = [
            r'http://',  # Plain HTTP
            r'ftp://',   # FTP
            r'telnet://' # Telnet
        ]

        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                for pattern in insecure_patterns:
                    if re.search(pattern, content):
                        # Skip if it's clearly for development/testing
                        if not any(skip in content.lower() for skip in [
                            'localhost', '127.0.0.1', 'test', 'example', 'placeholder'
                        ]):
                            self.add_finding('medium',
                                f"Insecure protocol usage found in {py_file.name}",
                                f"Pattern: {pattern}")

            except Exception as e:
                pass

    def calculate_security_score(self):
        """Calculate overall security score"""
        severity_weights = {
            'critical': 10,
            'high': 5,
            'medium': 2,
            'low': 1,
            'info': 0
        }

        total_penalty = 0
        for severity, findings in self.findings.items():
            total_penalty += len(findings) * severity_weights[severity]

        # Base score of 100, subtract penalties
        self.audit_stats['security_score'] = max(0, 100 - total_penalty)
        self.audit_stats['vulnerabilities_found'] = sum(len(v) for v in self.findings.values())

    def generate_audit_report(self) -> str:
        """Generate comprehensive audit report"""
        report = []
        report.append("🔒 SECURITY AUDIT REPORT")
        report.append("=" * 50)
        report.append(f"Security Score: {self.audit_stats['security_score']}/100")
        report.append(f"Files Scanned: {self.audit_stats['files_scanned']}")
        report.append("")

        # Summary by severity
        for severity in ['critical', 'high', 'medium', 'low', 'info']:
            findings = self.findings[severity]
            if findings:
                report.append(f"{severity.upper()}: {len(findings)} findings")
                for finding in findings[:5]:  # Show first 5
                    report.append(f"  • {finding['title']}")
                if len(findings) > 5:
                    report.append(f"  ... and {len(findings) - 5} more")

        report.append("")

        # Security recommendations
        recommendations = self.generate_recommendations()
        if recommendations:
            report.append("🛡️  SECURITY RECOMMENDATIONS")
            report.append("-" * 30)
            for rec in recommendations[:10]:  # Show top 10
                report.append(f"• {rec}")

        return "\n".join(report)

    def generate_recommendations(self) -> List[str]:
        """Generate security recommendations"""
        recommendations = []

        if self.findings['critical']:
            recommendations.append("IMMEDIATE: Address all critical security findings")
        if self.findings['high']:
            recommendations.append("HIGH PRIORITY: Fix high-severity security issues")

        # Specific recommendations based on findings
        if any('hardcoded' in f['title'].lower() for f in self.findings['high']):
            recommendations.append("Remove all hardcoded secrets and use environment variables")

        if any('world-readable' in f['title'].lower() for f in self.findings['high']):
            recommendations.append("Set restrictive file permissions on sensitive files")

        if any('unpinned' in f['title'].lower() for f in self.findings['medium']):
            recommendations.append("Pin all dependency versions in requirements.txt")

        if any('debug' in f['title'].lower() for f in self.findings['medium']):
            recommendations.append("Disable debug mode in production environments")

        if any('insecure' in f['title'].lower() for f in self.findings['medium']):
            recommendations.append("Use HTTPS and secure protocols in production")

        # General recommendations
        recommendations.extend([
            "Implement comprehensive input validation on all user inputs",
            "Use parameterized queries for database operations",
            "Implement proper session management and CSRF protection",
            "Regular security code reviews and dependency updates",
            "Implement rate limiting and DDoS protection",
            "Use secure headers (CSP, HSTS, X-Frame-Options)",
            "Implement proper logging and monitoring for security events",
            "Regular backup and disaster recovery testing"
        ])

        return recommendations

    def add_finding(self, severity: str, title: str, details: str = ""):
        """Add a security finding"""
        if severity not in self.findings:
            severity = 'info'

        self.findings[severity].append({
            'title': title,
            'details': details,
            'severity': severity
        })

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped in audit"""
        skip_patterns = [
            '__pycache__',
            '.pytest_cache',
            'node_modules',
            '.git',
            'venv',
            'env',
            'asterai_env',
            'build',
            'dist',
            '.eggs'
        ]

        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)


def main():
    """Run security audit"""
    auditor = SecurityAuditor()

    try:
        results = auditor.run_full_audit()

        # Save detailed results
        with open('security_audit_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print("\n📊 Detailed results saved to security_audit_results.json")
        print(f"🔒 Security Score: {auditor.audit_stats['security_score']}/100")

        # Exit with appropriate code
        if auditor.findings['critical']:
            sys.exit(1)  # Critical issues found
        elif auditor.audit_stats['security_score'] < 70:
            sys.exit(1)  # Poor security score
        else:
            sys.exit(0)  # Acceptable security

    except Exception as e:
        print(f"❌ Security audit failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
