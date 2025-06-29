#!/usr/bin/env python3
"""
Automated Code Quality Enhancement System for SLE TAGT Project

This script provides comprehensive automation for:
1. Code quality analysis and improvements
2. Automated testing and validation
3. Documentation generation
4. Performance monitoring
5. Security scanning
6. Dependency management
7. GPU optimization checks

Designed for Windows environment with GPU acceleration.
"""

import os
import sys
import subprocess
import json
import logging
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import platform
import psutil
import GPUtil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('code_quality_automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CodeQualityAutomator:
    """Comprehensive code quality automation system."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.results = {}
        self.system_info = self._get_system_info()
        self.gpu_info = self._get_gpu_info()
        
        # Create quality reports directory
        self.reports_dir = self.project_root / "quality_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized CodeQualityAutomator for {self.project_root}")
        logger.info(f"System: {self.system_info['os']} {self.system_info['version']}")
        logger.info(f"Python: {self.system_info['python_version']}")
        logger.info(f"GPU: {self.gpu_info['gpu_available']}")
    
    def _get_system_info(self) -> Dict:
        """Get system information for optimization."""
        return {
            'os': platform.system(),
            'version': platform.version(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'disk_space_gb': round(shutil.disk_usage('/').free / (1024**3), 2)
        }
    
    def _get_gpu_info(self) -> Dict:
        """Get GPU information for CUDA optimization."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Primary GPU
                return {
                    'gpu_available': True,
                    'gpu_name': gpu.name,
                    'gpu_memory_gb': round(gpu.memoryTotal / 1024, 2),
                    'gpu_utilization': gpu.load * 100,
                    'gpu_memory_used': round(gpu.memoryUsed / 1024, 2)
                }
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")
        
        return {'gpu_available': False}
    
    def install_quality_tools(self) -> bool:
        """Install code quality tools if not present."""
        tools = [
            'black',  # Code formatting
            'isort',  # Import sorting
            'flake8',  # Linting
            'mypy',   # Type checking
            'bandit', # Security scanning
            'pytest', # Testing
            'pytest-cov',  # Coverage
            'sphinx', # Documentation
            'pre-commit',  # Git hooks
            'safety',  # Dependency security
            'vulture', # Dead code detection
            'radon',   # Code complexity
            'autopep8' # Auto PEP8 formatting
        ]
        
        logger.info("Installing code quality tools...")
        try:
            for tool in tools:
                subprocess.run([sys.executable, '-m', 'pip', 'install', tool], 
                             check=True, capture_output=True)
            logger.info("All quality tools installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install tools: {e}")
            return False
    
    def format_code(self) -> Dict:
        """Automatically format Python code."""
        logger.info("Formatting code with black and isort...")
        results = {'black': False, 'isort': False, 'autopep8': False}
        
        try:
            # Black formatting
            black_result = subprocess.run(
                ['python', '-m', 'black', '--line-length', '88', str(self.project_root / 'src')],
                capture_output=True, text=True
            )
            results['black'] = black_result.returncode == 0
            
            # Import sorting
            isort_result = subprocess.run(
                ['python', '-m', 'isort', str(self.project_root / 'src')],
                capture_output=True, text=True
            )
            results['isort'] = isort_result.returncode == 0
            
            # AutoPEP8 for additional formatting
            autopep8_result = subprocess.run(
                ['python', '-m', 'autopep8', '--in-place', '--recursive', 
                 str(self.project_root / 'src')],
                capture_output=True, text=True
            )
            results['autopep8'] = autopep8_result.returncode == 0
            
            logger.info(f"Code formatting results: {results}")
            
        except Exception as e:
            logger.error(f"Code formatting failed: {e}")
        
        return results
    
    def run_linting(self) -> Dict:
        """Run comprehensive linting analysis."""
        logger.info("Running linting analysis...")
        results = {}
        
        try:
            # Flake8 linting
            flake8_result = subprocess.run(
                ['python', '-m', 'flake8', str(self.project_root / 'src'), 
                 '--max-line-length=88', '--extend-ignore=E203,W503'],
                capture_output=True, text=True
            )
            results['flake8'] = {
                'success': flake8_result.returncode == 0,
                'output': flake8_result.stdout,
                'errors': flake8_result.stderr
            }
            
            # Save flake8 report
            with open(self.reports_dir / 'flake8_report.txt', 'w') as f:
                f.write(flake8_result.stdout)
            
            logger.info(f"Flake8 completed with return code: {flake8_result.returncode}")
            
        except Exception as e:
            logger.error(f"Linting failed: {e}")
            results['flake8'] = {'success': False, 'error': str(e)}
        
        return results
    
    def run_type_checking(self) -> Dict:
        """Run mypy type checking."""
        logger.info("Running type checking with mypy...")
        results = {}
        
        try:
            mypy_result = subprocess.run(
                ['python', '-m', 'mypy', str(self.project_root / 'src'), 
                 '--ignore-missing-imports', '--strict-optional'],
                capture_output=True, text=True
            )
            
            results['mypy'] = {
                'success': mypy_result.returncode == 0,
                'output': mypy_result.stdout,
                'errors': mypy_result.stderr
            }
            
            # Save mypy report
            with open(self.reports_dir / 'mypy_report.txt', 'w') as f:
                f.write(mypy_result.stdout)
            
            logger.info(f"MyPy completed with return code: {mypy_result.returncode}")
            
        except Exception as e:
            logger.error(f"Type checking failed: {e}")
            results['mypy'] = {'success': False, 'error': str(e)}
        
        return results
    
    def run_security_scan(self) -> Dict:
        """Run security vulnerability scanning."""
        logger.info("Running security scans...")
        results = {}
        
        try:
            # Bandit security scan
            bandit_result = subprocess.run(
                ['python', '-m', 'bandit', '-r', str(self.project_root / 'src'), 
                 '-f', 'json', '-o', str(self.reports_dir / 'bandit_report.json')],
                capture_output=True, text=True
            )
            
            results['bandit'] = {
                'success': bandit_result.returncode in [0, 1],  # 1 is issues found
                'output': bandit_result.stdout,
                'errors': bandit_result.stderr
            }
            
            # Safety check for dependencies
            safety_result = subprocess.run(
                ['python', '-m', 'safety', 'check', '--json'],
                capture_output=True, text=True
            )
            
            results['safety'] = {
                'success': safety_result.returncode == 0,
                'output': safety_result.stdout,
                'errors': safety_result.stderr
            }
            
            # Save safety report
            with open(self.reports_dir / 'safety_report.json', 'w') as f:
                f.write(safety_result.stdout)
            
            logger.info("Security scans completed")
            
        except Exception as e:
            logger.error(f"Security scanning failed: {e}")
            results['security_error'] = str(e)
        
        return results
    
    def analyze_code_complexity(self) -> Dict:
        """Analyze code complexity and maintainability."""
        logger.info("Analyzing code complexity...")
        results = {}
        
        try:
            # Radon complexity analysis
            radon_cc_result = subprocess.run(
                ['python', '-m', 'radon', 'cc', str(self.project_root / 'src'), 
                 '--json'],
                capture_output=True, text=True
            )
            
            radon_mi_result = subprocess.run(
                ['python', '-m', 'radon', 'mi', str(self.project_root / 'src'), 
                 '--json'],
                capture_output=True, text=True
            )
            
            results['complexity'] = {
                'cyclomatic': json.loads(radon_cc_result.stdout) if radon_cc_result.stdout else {},
                'maintainability': json.loads(radon_mi_result.stdout) if radon_mi_result.stdout else {}
            }
            
            # Save complexity reports
            with open(self.reports_dir / 'complexity_report.json', 'w') as f:
                json.dump(results['complexity'], f, indent=2)
            
            logger.info("Code complexity analysis completed")
            
        except Exception as e:
            logger.error(f"Complexity analysis failed: {e}")
            results['complexity_error'] = str(e)
        
        return results
    
    def detect_dead_code(self) -> Dict:
        """Detect unused code with vulture."""
        logger.info("Detecting dead code...")
        results = {}
        
        try:
            vulture_result = subprocess.run(
                ['python', '-m', 'vulture', str(self.project_root / 'src')],
                capture_output=True, text=True
            )
            
            results['dead_code'] = {
                'success': True,
                'output': vulture_result.stdout,
                'errors': vulture_result.stderr
            }
            
            # Save dead code report
            with open(self.reports_dir / 'dead_code_report.txt', 'w') as f:
                f.write(vulture_result.stdout)
            
            logger.info("Dead code detection completed")
            
        except Exception as e:
            logger.error(f"Dead code detection failed: {e}")
            results['dead_code_error'] = str(e)
        
        return results
    
    def run_tests_with_coverage(self) -> Dict:
        """Run tests with coverage analysis."""
        logger.info("Running tests with coverage...")
        results = {}
        
        try:
            # Run pytest with coverage
            pytest_result = subprocess.run([
                'python', '-m', 'pytest', 
                str(self.project_root / 'tests'),
                '--cov=' + str(self.project_root / 'src'),
                '--cov-report=html:' + str(self.reports_dir / 'coverage_html'),
                '--cov-report=json:' + str(self.reports_dir / 'coverage.json'),
                '--cov-report=term',
                '-v'
            ], capture_output=True, text=True)
            
            results['tests'] = {
                'success': pytest_result.returncode == 0,
                'output': pytest_result.stdout,
                'errors': pytest_result.stderr
            }
            
            logger.info(f"Tests completed with return code: {pytest_result.returncode}")
            
        except Exception as e:
            logger.error(f"Testing failed: {e}")
            results['tests_error'] = str(e)
        
        return results
    
    def optimize_for_gpu(self) -> Dict:
        """Check and optimize GPU usage."""
        logger.info("Checking GPU optimization...")
        results = {'gpu_optimizations': []}
        
        if not self.gpu_info['gpu_available']:
            results['gpu_optimizations'].append("No GPU detected - consider CPU optimizations")
            return results
        
        # Check PyTorch CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                results['cuda_available'] = True
                results['cuda_device_count'] = torch.cuda.device_count()
                results['cuda_device_name'] = torch.cuda.get_device_name(0)
                
                # GPU memory optimization suggestions
                if self.gpu_info['gpu_memory_gb'] < 8:
                    results['gpu_optimizations'].append(
                        "Consider reducing batch size for low GPU memory"
                    )
                
                if self.gpu_info['gpu_utilization'] < 50:
                    results['gpu_optimizations'].append(
                        "GPU utilization is low - consider increasing batch size"
                    )
                
            else:
                results['cuda_available'] = False
                results['gpu_optimizations'].append(
                    "CUDA not available - install CUDA-enabled PyTorch"
                )
                
        except ImportError:
            results['gpu_optimizations'].append("PyTorch not installed")
        
        return results
    
    def generate_documentation(self) -> Dict:
        """Generate project documentation."""
        logger.info("Generating documentation...")
        results = {}
        
        try:
            # Create docs directory
            docs_dir = self.project_root / "docs" / "auto_generated"
            docs_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate API documentation with sphinx
            sphinx_result = subprocess.run([
                'python', '-m', 'sphinx.cmd.quickstart',
                '--quiet', '--project=SLE-TAGT', '--author=AutoGen',
                '--release=1.0', '--language=en', '--makefile',
                str(docs_dir)
            ], capture_output=True, text=True)
            
            results['documentation'] = {
                'success': sphinx_result.returncode == 0,
                'output': sphinx_result.stdout,
                'errors': sphinx_result.stderr
            }
            
            logger.info("Documentation generation completed")
            
        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            results['documentation_error'] = str(e)
        
        return results
    
    def create_quality_dashboard(self) -> None:
        """Create an HTML dashboard with all quality metrics."""
        logger.info("Creating quality dashboard...")
        
        dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SLE TAGT - Code Quality Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .success {{ background: #d4edda; border-color: #c3e6cb; }}
        .warning {{ background: #fff3cd; border-color: #ffeaa7; }}
        .error {{ background: #f8d7da; border-color: #f5c6cb; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 3px; }}
        .gpu-info {{ background: #e7f3ff; padding: 10px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 SLE TAGT Code Quality Dashboard</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section gpu-info">
        <h2>🖥️ System Information</h2>
        <div class="metric">OS: {self.system_info['os']} {self.system_info['version']}</div>
        <div class="metric">Python: {self.system_info['python_version']}</div>
        <div class="metric">CPU Cores: {self.system_info['cpu_count']}</div>
        <div class="metric">Memory: {self.system_info['memory_gb']} GB</div>
        <div class="metric">GPU: {'✅ Available' if self.gpu_info['gpu_available'] else '❌ Not Available'}</div>
        {f'<div class="metric">GPU: {self.gpu_info["gpu_name"]} ({self.gpu_info["gpu_memory_gb"]} GB)</div>' if self.gpu_info['gpu_available'] else ''}
    </div>
    
    <div class="section">
        <h2>📊 Quality Metrics Summary</h2>
        <p>Comprehensive analysis results will be displayed here after running all checks.</p>
        <ul>
            <li>✅ Code Formatting: Automated with Black, isort, and autopep8</li>
            <li>🔍 Linting: Flake8 analysis for code quality</li>
            <li>🛡️ Security: Bandit and Safety vulnerability scanning</li>
            <li>📈 Complexity: Radon cyclomatic complexity analysis</li>
            <li>🧪 Testing: Pytest with coverage reporting</li>
            <li>📚 Documentation: Auto-generated API docs</li>
            <li>⚡ GPU Optimization: CUDA utilization analysis</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>🎯 Recommendations</h2>
        <ul>
            <li>Run automated formatting before each commit</li>
            <li>Maintain test coverage above 80%</li>
            <li>Keep cyclomatic complexity below 10</li>
            <li>Regular security dependency updates</li>
            <li>Optimize GPU memory usage for training</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>📁 Generated Reports</h2>
        <ul>
            <li><a href="flake8_report.txt">Flake8 Linting Report</a></li>
            <li><a href="mypy_report.txt">MyPy Type Checking Report</a></li>
            <li><a href="bandit_report.json">Bandit Security Report</a></li>
            <li><a href="safety_report.json">Safety Dependency Report</a></li>
            <li><a href="complexity_report.json">Code Complexity Report</a></li>
            <li><a href="dead_code_report.txt">Dead Code Detection Report</a></li>
            <li><a href="coverage_html/index.html">Test Coverage Report</a></li>
        </ul>
    </div>
</body>
</html>
        """
        
        with open(self.reports_dir / 'quality_dashboard.html', 'w') as f:
            f.write(dashboard_html)
        
        logger.info(f"Quality dashboard created: {self.reports_dir / 'quality_dashboard.html'}")
    
    def run_full_automation(self) -> Dict:
        """Run complete automated code quality enhancement."""
        logger.info("🚀 Starting full code quality automation...")
        start_time = time.time()
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.system_info,
            'gpu_info': self.gpu_info
        }
        
        # Step 1: Install tools
        logger.info("Step 1: Installing quality tools...")
        all_results['tool_installation'] = self.install_quality_tools()
        
        # Step 2: Format code
        logger.info("Step 2: Formatting code...")
        all_results['formatting'] = self.format_code()
        
        # Step 3: Run linting
        logger.info("Step 3: Running linting...")
        all_results['linting'] = self.run_linting()
        
        # Step 4: Type checking
        logger.info("Step 4: Type checking...")
        all_results['type_checking'] = self.run_type_checking()
        
        # Step 5: Security scanning
        logger.info("Step 5: Security scanning...")
        all_results['security'] = self.run_security_scan()
        
        # Step 6: Complexity analysis
        logger.info("Step 6: Analyzing complexity...")
        all_results['complexity'] = self.analyze_code_complexity()
        
        # Step 7: Dead code detection
        logger.info("Step 7: Detecting dead code...")
        all_results['dead_code'] = self.detect_dead_code()
        
        # Step 8: Run tests
        logger.info("Step 8: Running tests with coverage...")
        all_results['testing'] = self.run_tests_with_coverage()
        
        # Step 9: GPU optimization
        logger.info("Step 9: GPU optimization analysis...")
        all_results['gpu_optimization'] = self.optimize_for_gpu()
        
        # Step 10: Generate documentation
        logger.info("Step 10: Generating documentation...")
        all_results['documentation'] = self.generate_documentation()
        
        # Step 11: Create dashboard
        logger.info("Step 11: Creating quality dashboard...")
        self.create_quality_dashboard()
        
        # Save comprehensive results
        all_results['execution_time'] = time.time() - start_time
        
        with open(self.reports_dir / 'comprehensive_quality_report.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"✅ Full automation completed in {all_results['execution_time']:.2f} seconds")
        logger.info(f"📊 Reports saved to: {self.reports_dir}")
        logger.info(f"🌐 Dashboard: {self.reports_dir / 'quality_dashboard.html'}")
        
        return all_results

def main():
    """Main execution function."""
    print("🚀 SLE TAGT Code Quality Automation System")
    print("=" * 50)
    
    # Initialize automator
    automator = CodeQualityAutomator()
    
    # Run full automation
    results = automator.run_full_automation()
    
    print("\n✅ Automation Complete!")
    print(f"📊 Check reports in: {automator.reports_dir}")
    print(f"🌐 Open dashboard: {automator.reports_dir / 'quality_dashboard.html'}")
    
    return results

if __name__ == "__main__":
    main()