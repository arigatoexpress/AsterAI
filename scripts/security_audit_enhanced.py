#!/usr/bin/env python3
"""
Enhanced Security Audit Tool for AI Trading System

Comprehensive security assessment covering:
- API key exposure and secrets management
- File permissions and access controls
- Input validation and sanitization
- Network security (SSL/TLS, certificates)
- Code security (TODO/FIXME analysis)
- Git history security scan
"""

import os
import sys
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
import hashlib
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class SecurityAuditor:
    """Comprehensive security audit tool"""

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.findings = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': [],
            'info': []
        }

        # Security patterns
        self.secret_patterns = [
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'api_secret\s*=\s*["\'][^"\']+["\']',
            r'password\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'private_key\s*=\s*["\'][^"\']+["\']',
            r'AWS_ACCESS_KEY_ID\s*=\s*["\'][^"\']+["\']',
            r'AWS_SECRET_ACCESS_KEY\s*=\s*["\'][^"\']+["\']',
            r'DATABASE_URL\s*=\s*["\'][^"\']+["\']',
        ]

        self.vulnerable_imports = [
            'pickle', 'eval', 'exec', 'subprocess.call',
            'os.system', 'os.popen', 'shell=True'
        ]

    def add_finding(self, severity: str, category: str, message: str, file_path: str = None, line: int = None):
        """Add a security finding"""
        finding = {
            'category': category,
            'message': message,
            'file': str(file_path) if file_path else None,
            'line': line,
            'timestamp': datetime.now().isoformat()
        }
        self.findings[severity].append(finding)

    def scan_hardcoded_secrets(self) -> None:
        """Scan for hardcoded secrets and API keys"""
        print("🔐 Scanning for hardcoded secrets...")

        # Skip certain directories
        skip_dirs = {'__pycache__', '.git', 'archive', 'asterai_env', 'asterai_env_fresh'}

        for root, dirs, files in os.walk(self.root_path):
            # Remove skipped directories
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for file in files:
                if file.endswith(('.py', '.json', '.yaml', '.yml', '.env', '.cfg', '.ini')):
                    file_path = Path(root) / file

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        for pattern in self.secret_patterns:
                            matches = re.finditer(pattern, content, re.IGNORECASE)
                            for match in matches:
                                line_num = content[:match.start()].count('\n') + 1

                                # Skip false positives (comments, test files, example files)
                                line = content.split('\n')[line_num - 1].strip()
                                if (line.startswith('#') or
                                    'example' in str(file_path).lower() or
                                    'template' in str(file_path).lower() or
                                    'test' in str(file_path).lower() or
                                    'mock' in str(file_path).lower()):
                                    continue

                                self.add_finding(
                                    'critical',
                                    'HARDCODED_SECRET',
                                    f'Potential hardcoded secret found: {match.group()[:50]}...',
                                    file_path,
                                    line_num
                                )

                    except (UnicodeDecodeError, IOError):
                        continue

    def check_file_permissions(self) -> None:
        """Check file permissions for sensitive files"""
        print("🔒 Checking file permissions...")

        sensitive_files = [
            '.env', '.secrets.json', '.api_keys.json',
            'config/secrets.json', 'config/api_keys.json'
        ]

        for sensitive_file in sensitive_files:
            file_path = self.root_path / sensitive_file
            if file_path.exists():
                # On Windows, check if file is hidden or has restrictive permissions
                try:
                    # Check if file is in .gitignore
                    gitignore_path = self.root_path / '.gitignore'
                    if gitignore_path.exists():
                        with open(gitignore_path, 'r') as f:
                            gitignore_content = f.read()

                        if sensitive_file not in gitignore_content:
                            self.add_finding(
                                'high',
                                'GITIGNORE_MISSING',
                                f'Sensitive file {sensitive_file} not in .gitignore',
                                file_path
                            )

                except Exception:
                    pass

    def scan_git_history(self) -> None:
        """Scan git history for exposed credentials"""
        print("📚 Scanning git history for exposed credentials...")

        try:
            # Check if we're in a git repository
            result = subprocess.run(['git', 'log', '--oneline', '-10'],
                                  capture_output=True, text=True, cwd=self.root_path,
                                  encoding='utf-8', errors='ignore')

            if result.returncode == 0:
                # Look for commits that might contain secrets
                result = subprocess.run(['git', 'log', '--all', '--full-history',
                                       '--', '*.py', '*.json', '*.env'],
                                      capture_output=True, text=True, cwd=self.root_path,
                                      encoding='utf-8', errors='ignore')

                stdout = result.stdout or ""
                if 'api_key' in stdout.lower() or 'secret' in stdout.lower():
                    self.add_finding(
                        'high',
                        'GIT_HISTORY_CHECK',
                        'Git history may contain exposed credentials - manual review recommended'
                    )

        except (subprocess.CalledProcessError, FileNotFoundError, UnicodeDecodeError):
            self.add_finding(
                'info',
                'GIT_CHECK_FAILED',
                'Could not check git history for exposed credentials'
            )

    def scan_input_validation(self) -> None:
        """Scan for input validation and sanitization"""
        print("🛡️  Scanning for input validation...")

        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if d not in {'__pycache__', '.git', 'archive'}]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # Check for dangerous patterns
                        dangerous_patterns = [
                            (r'input\s*\(', 'HIGH', 'Raw input() usage found'),
                            (r'eval\s*\(', 'CRITICAL', 'Dangerous eval() usage found'),
                            (r'exec\s*\(', 'CRITICAL', 'Dangerous exec() usage found'),
                            (r'subprocess\..*shell\s*=\s*True', 'HIGH', 'Shell=True in subprocess calls'),
                            (r'os\.system\s*\(', 'HIGH', 'os.system() usage found'),
                            (r'pickle\.loads?\s*\(', 'MEDIUM', 'Pickle usage found (security risk)'),
                        ]

                        for pattern, severity, message in dangerous_patterns:
                            matches = re.finditer(pattern, content)
                            for match in matches:
                                line_num = content[:match.start()].count('\n') + 1
                                self.add_finding(
                                    severity.lower(),
                                    'CODE_SECURITY',
                                    message,
                                    file_path,
                                    line_num
                                )

                        # Check for SQL injection patterns
                        sql_patterns = [
                            r'execute\s*\(\s*["\'].*\%.*["\']',
                            r'execute\s*\(\s*["\'].*\+.*["\']',
                            r'cursor\.execute\s*\(\s*.*format\s*\(',
                        ]

                        for pattern in sql_patterns:
                            if re.search(pattern, content):
                                self.add_finding(
                                    'high',
                                    'SQL_INJECTION',
                                    'Potential SQL injection vulnerability',
                                    file_path
                                )

                    except (UnicodeDecodeError, IOError):
                        continue

    def scan_network_security(self) -> None:
        """Scan for network security issues"""
        print("🌐 Scanning network security...")

        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if d not in {'__pycache__', '.git', 'archive'}]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # Check for HTTP URLs (should be HTTPS)
                        http_urls = re.findall(r'http://[^\s"\']+', content)
                        for url in http_urls:
                            if 'localhost' not in url and '127.0.0.1' not in url:
                                self.add_finding(
                                    'medium',
                                    'INSECURE_HTTP',
                                    f'Insecure HTTP URL found: {url}',
                                    file_path
                                )

                        # Check for requests without verify=False (but not localhost)
                        if 'requests.' in content and 'verify=False' in content:
                            self.add_finding(
                                'high',
                                'SSL_VERIFICATION_DISABLED',
                                'SSL certificate verification disabled',
                                file_path
                            )

                        # Check for API endpoints without authentication
                        api_patterns = [
                            r'@app\.route\s*\(',
                            r'fastapi\.APIRouter',
                            r'def\s+\w+.*request.*\):',
                        ]

                        has_auth = 'auth' in content.lower() or 'token' in content.lower() or 'jwt' in content.lower()

                        for pattern in api_patterns:
                            if re.search(pattern, content) and not has_auth:
                                self.add_finding(
                                    'medium',
                                    'MISSING_AUTHENTICATION',
                                    'API endpoint without apparent authentication',
                                    file_path
                                )

                    except (UnicodeDecodeError, IOError):
                        continue

    def scan_todo_comments(self) -> None:
        """Scan for TODO/FIXME comments that may indicate security issues"""
        print("📝 Scanning for security-related TODO comments...")

        security_keywords = [
            'security', 'auth', 'encrypt', 'decrypt', 'hash', 'password',
            'token', 'key', 'secret', 'vulnerability', 'exploit', 'hack',
            'sanitize', 'validate', 'escape', 'csrf', 'xss', 'injection'
        ]

        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if d not in {'__pycache__', '.git', 'archive'}]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()

                        for i, line in enumerate(lines, 1):
                            lower_line = line.lower().strip()
                            if ('todo' in lower_line or 'fixme' in lower_line or 'hack' in lower_line):
                                # Check if it contains security keywords
                                if any(keyword in lower_line for keyword in security_keywords):
                                    self.add_finding(
                                        'medium',
                                        'SECURITY_TODO',
                                        f'Security-related TODO/FIXME found: {line.strip()[:100]}',
                                        file_path,
                                        i
                                    )

                    except (UnicodeDecodeError, IOError):
                        continue

    def generate_report(self) -> str:
        """Generate security audit report"""
        lines = []
        lines.append("="*80)
        lines.append("ENHANCED SECURITY AUDIT REPORT")
        lines.append("="*80)
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Root Directory: {self.root_path}")
        lines.append("")

        # Summary
        total_findings = sum(len(findings) for findings in self.findings.values())
        lines.append("📊 SUMMARY")
        lines.append(f"   Total findings: {total_findings}")
        for severity in ['critical', 'high', 'medium', 'low', 'info']:
            count = len(self.findings[severity])
            if count > 0:
                lines.append(f"   {severity.upper()}: {count}")
        lines.append("")

        # Detailed findings
        severity_order = ['critical', 'high', 'medium', 'low', 'info']
        severity_icons = {
            'critical': '🚨',
            'high': '🔴',
            'medium': '🟡',
            'low': '🟢',
            'info': 'ℹ️'
        }

        for severity in severity_order:
            findings = self.findings[severity]
            if findings:
                lines.append(f"{severity_icons[severity]} {severity.upper()} SEVERITY ({len(findings)})")
                lines.append("-" * 50)

                for finding in findings:
                    if finding['file']:
                        lines.append(f"File: {finding['file']}")
                        if finding['line']:
                            lines.append(f"Line: {finding['line']}")
                    lines.append(f"Category: {finding['category']}")
                    lines.append(f"Message: {finding['message']}")
                    lines.append("")

        # Recommendations
        lines.append("✅ RECOMMENDATIONS")
        lines.append("-" * 50)
        lines.append("1. IMMEDIATE ACTION REQUIRED:")
        if self.findings['critical']:
            lines.append("   - Address all CRITICAL findings immediately")
        if self.findings['high']:
            lines.append("   - Review and fix HIGH severity issues")
        lines.append("")
        lines.append("2. SECRETS MANAGEMENT:")
        lines.append("   - Move all hardcoded secrets to environment variables")
        lines.append("   - Use .env files for local development")
        lines.append("   - Use GCP Secret Manager for production")
        lines.append("   - Ensure sensitive files are in .gitignore")
        lines.append("")
        lines.append("3. CODE SECURITY:")
        lines.append("   - Replace dangerous functions (eval, exec, pickle)")
        lines.append("   - Implement proper input validation")
        lines.append("   - Use parameterized queries for database operations")
        lines.append("")
        lines.append("4. NETWORK SECURITY:")
        lines.append("   - Use HTTPS for all external API calls")
        lines.append("   - Enable SSL certificate verification")
        lines.append("   - Implement proper authentication for API endpoints")

        return "\n".join(lines)

    def run_full_audit(self) -> None:
        """Run the complete security audit"""
        print("🔒 STARTING ENHANCED SECURITY AUDIT")
        print("="*50)

        # Run all security checks
        self.scan_hardcoded_secrets()
        self.check_file_permissions()
        self.scan_git_history()
        self.scan_input_validation()
        self.scan_network_security()
        self.scan_todo_comments()

        # Generate and save report
        report = self.generate_report()
        print("\n" + report)

        # Save detailed JSON report
        json_report = {
            'timestamp': datetime.now().isoformat(),
            'root_directory': str(self.root_path),
            'findings': self.findings,
            'summary': {
                severity: len(findings)
                for severity, findings in self.findings.items()
            }
        }

        report_path = Path("SECURITY_AUDIT_REPORT.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, default=str)

        print(f"\n✅ Detailed JSON report saved to: {report_path}")

        # Save text report
        text_report_path = Path("SECURITY_AUDIT_REPORT.md")
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✅ Text report saved to: {text_report_path}")


def main():
    """Main security audit function"""
    root_path = Path(__file__).parent.parent
    auditor = SecurityAuditor(root_path)
    auditor.run_full_audit()


if __name__ == "__main__":
    main()
