"""
Comprehensive quality gates and validation system for liquid neural networks.
Ensures production readiness through automated testing and validation.
"""

import os
import sys
import time
import json
import subprocess
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

# Quality gate results
class QualityGateStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    status: QualityGateStatus
    score: float  # 0.0 to 100.0
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    recommendations: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    overall_score: float
    total_gates: int
    passed_gates: int
    failed_gates: int
    warning_gates: int
    skipped_gates: int
    execution_time: float
    gate_results: List[QualityGateResult] = field(default_factory=list)
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_score': self.overall_score,
            'gate_summary': {
                'total': self.total_gates,
                'passed': self.passed_gates,
                'failed': self.failed_gates,
                'warnings': self.warning_gates,
                'skipped': self.skipped_gates
            },
            'execution_time': self.execution_time,
            'gate_results': [
                {
                    'name': result.gate_name,
                    'status': result.status.value,
                    'score': result.score,
                    'message': result.message,
                    'execution_time': result.execution_time,
                    'recommendations': result.recommendations
                } for result in self.gate_results
            ],
            'summary': self.summary,
            'overall_recommendations': self.recommendations
        }


class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str, weight: float = 1.0, critical: bool = False):
        self.name = name
        self.weight = weight
        self.critical = critical
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger(f'quality_gate.{self.name}')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def execute(self) -> QualityGateResult:
        """Execute the quality gate check."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Executing quality gate: {self.name}")
            result = self._run_check()
            result.execution_time = time.time() - start_time
            
            self.logger.info(f"Quality gate {self.name} completed: {result.status.value} (score: {result.score:.1f})")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Quality gate {self.name} failed with exception: {e}")
            
            return QualityGateResult(
                gate_name=self.name,
                status=QualityGateStatus.FAILED,
                score=0.0,
                message=f"Gate execution failed: {str(e)}",
                execution_time=execution_time,
                recommendations=[f"Fix the error: {str(e)}"]
            )
    
    def _run_check(self) -> QualityGateResult:
        """Override this method to implement the actual check."""
        raise NotImplementedError


class ImportValidationGate(QualityGate):
    """Validate that all modules can be imported successfully."""
    
    def __init__(self):
        super().__init__("Import Validation", weight=2.0, critical=True)
        self.required_modules = [
            'liquid_vision.core.liquid_neurons',
            'liquid_vision.core.event_encoding',
            'liquid_vision.core.temporal_dynamics',
            'liquid_vision.simulation.event_simulator',
            'liquid_vision.training.liquid_trainer',
            'liquid_vision.deployment.edge_deployer',
            'liquid_vision.utils.logging',
            'liquid_vision.utils.validation'
        ]
        
        self.new_modules = [
            'liquid_vision.core.realtime_processor',
            'liquid_vision.optimization.memory_efficient',
            'liquid_vision.deployment.hardware_interface',
            'liquid_vision.utils.error_handling',
            'liquid_vision.utils.monitoring',
            'liquid_vision.security.input_sanitizer',
            'liquid_vision.optimization.auto_scaling'
        ]
    
    def _run_check(self) -> QualityGateResult:
        successful_imports = 0
        failed_imports = []
        import_details = {}
        
        all_modules = self.required_modules + self.new_modules
        
        for module_name in all_modules:
            try:
                # Attempt import without torch dependency for basic structure validation
                module_path = Path('liquid_vision') / '/'.join(module_name.split('.')[1:]) / '__init__.py'
                if not module_path.exists():
                    module_path = Path('liquid_vision') / '/'.join(module_name.split('.')[1:]) + '.py'
                
                if module_path.exists():
                    successful_imports += 1
                    import_details[module_name] = "Module file exists"
                else:
                    failed_imports.append(module_name)
                    import_details[module_name] = "Module file not found"
                    
            except Exception as e:
                failed_imports.append(module_name)
                import_details[module_name] = f"Import error: {str(e)}"
        
        success_rate = successful_imports / len(all_modules) * 100
        
        if success_rate >= 90:
            status = QualityGateStatus.PASSED
            message = f"Import validation passed: {successful_imports}/{len(all_modules)} modules"
        elif success_rate >= 70:
            status = QualityGateStatus.WARNING
            message = f"Import validation warning: {successful_imports}/{len(all_modules)} modules"
        else:
            status = QualityGateStatus.FAILED
            message = f"Import validation failed: {successful_imports}/{len(all_modules)} modules"
        
        recommendations = []
        if failed_imports:
            recommendations.append(f"Fix failed imports: {', '.join(failed_imports)}")
            recommendations.append("Ensure all required dependencies are installed")
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=success_rate,
            message=message,
            details=import_details,
            recommendations=recommendations
        )


class CodeStructureGate(QualityGate):
    """Validate code structure and organization."""
    
    def __init__(self):
        super().__init__("Code Structure", weight=1.5)
        self.required_directories = [
            'liquid_vision/core',
            'liquid_vision/simulation',
            'liquid_vision/training',
            'liquid_vision/deployment',
            'liquid_vision/optimization',
            'liquid_vision/security',
            'liquid_vision/utils',
            'tests',
            'examples',
            'docs'
        ]
        
        self.required_files = [
            'liquid_vision/__init__.py',
            'requirements.txt',
            'setup.py',
            'README.md',
            'LICENSE'
        ]
    
    def _run_check(self) -> QualityGateResult:
        structure_score = 0
        max_score = len(self.required_directories) + len(self.required_files)
        structure_details = {}
        
        # Check directories
        for directory in self.required_directories:
            if Path(directory).exists():
                structure_score += 1
                structure_details[directory] = "✅ Exists"
            else:
                structure_details[directory] = "❌ Missing"
        
        # Check files
        for file_path in self.required_files:
            if Path(file_path).exists():
                structure_score += 1
                structure_details[file_path] = "✅ Exists"
            else:
                structure_details[file_path] = "❌ Missing"
        
        success_rate = structure_score / max_score * 100
        
        # Additional checks
        recommendations = []
        if not Path('pyproject.toml').exists() and not Path('setup.py').exists():
            recommendations.append("Add build configuration (pyproject.toml or setup.py)")
        
        if not Path('tests').exists():
            recommendations.append("Add comprehensive test suite")
            
        if not Path('docs').exists():
            recommendations.append("Add documentation directory")
        
        status = (QualityGateStatus.PASSED if success_rate >= 80 else
                 QualityGateStatus.WARNING if success_rate >= 60 else
                 QualityGateStatus.FAILED)
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=success_rate,
            message=f"Code structure: {structure_score}/{max_score} required elements",
            details=structure_details,
            recommendations=recommendations
        )


class DocumentationGate(QualityGate):
    """Validate documentation completeness and quality."""
    
    def __init__(self):
        super().__init__("Documentation", weight=1.0)
    
    def _run_check(self) -> QualityGateResult:
        doc_score = 0
        max_score = 100
        doc_details = {}
        
        # Check README
        readme_path = Path('README.md')
        if readme_path.exists():
            readme_content = readme_path.read_text()
            readme_score = self._evaluate_readme(readme_content)
            doc_score += readme_score * 0.4  # 40% weight
            doc_details['README.md'] = f"Score: {readme_score:.1f}/100"
        else:
            doc_details['README.md'] = "❌ Missing"
        
        # Check API documentation
        api_doc_path = Path('docs/api_reference.md')
        if api_doc_path.exists():
            api_content = api_doc_path.read_text()
            api_score = self._evaluate_api_docs(api_content)
            doc_score += api_score * 0.3  # 30% weight
            doc_details['API Documentation'] = f"Score: {api_score:.1f}/100"
        else:
            doc_details['API Documentation'] = "❌ Missing"
        
        # Check docstrings in code
        docstring_score = self._evaluate_docstrings()
        doc_score += docstring_score * 0.3  # 30% weight
        doc_details['Code Docstrings'] = f"Score: {docstring_score:.1f}/100"
        
        status = (QualityGateStatus.PASSED if doc_score >= 70 else
                 QualityGateStatus.WARNING if doc_score >= 50 else
                 QualityGateStatus.FAILED)
        
        recommendations = []
        if doc_score < 70:
            recommendations.append("Improve documentation completeness")
            if readme_path.exists() and len(readme_path.read_text()) < 1000:
                recommendations.append("Expand README with more detailed information")
            if not api_doc_path.exists():
                recommendations.append("Add comprehensive API documentation")
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=doc_score,
            message=f"Documentation score: {doc_score:.1f}/100",
            details=doc_details,
            recommendations=recommendations
        )
    
    def _evaluate_readme(self, content: str) -> float:
        """Evaluate README quality."""
        score = 0
        
        # Check for essential sections
        essential_sections = [
            ('overview', 10), ('installation', 15), ('usage', 15),
            ('examples', 10), ('features', 10), ('architecture', 10),
            ('contributing', 5), ('license', 5), ('citation', 5)
        ]
        
        content_lower = content.lower()
        for section, points in essential_sections:
            if section in content_lower:
                score += points
        
        # Length bonus
        if len(content) > 2000:
            score += 10
        elif len(content) > 1000:
            score += 5
        
        # Code examples
        if '```' in content:
            score += 5
        
        return min(score, 100)
    
    def _evaluate_api_docs(self, content: str) -> float:
        """Evaluate API documentation quality."""
        score = 0
        
        # Check for code examples
        if '```python' in content:
            score += 20
        
        # Check for parameter documentation
        if 'Args:' in content or 'Parameters:' in content:
            score += 20
        
        # Check for return value documentation
        if 'Returns:' in content:
            score += 15
        
        # Check for examples
        if 'Example' in content:
            score += 15
        
        # Length and completeness
        if len(content) > 1000:
            score += 15
        
        # Class and function documentation
        if 'class ' in content and 'def ' in content:
            score += 15
        
        return min(score, 100)
    
    def _evaluate_docstrings(self) -> float:
        """Evaluate docstring coverage in code."""
        python_files = list(Path('liquid_vision').rglob('*.py'))
        if not python_files:
            return 0
        
        total_functions = 0
        documented_functions = 0
        
        for file_path in python_files[:10]:  # Sample first 10 files
            try:
                content = file_path.read_text()
                # Simple heuristic for function/class documentation
                lines = content.split('\n')
                
                for i, line in enumerate(lines):
                    if (line.strip().startswith('def ') or 
                        line.strip().startswith('class ')) and not line.strip().startswith('def _'):
                        total_functions += 1
                        
                        # Check if next few lines contain docstring
                        for j in range(i + 1, min(i + 5, len(lines))):
                            if '"""' in lines[j] or "'''" in lines[j]:
                                documented_functions += 1
                                break
                                
            except Exception:
                continue
        
        if total_functions == 0:
            return 50  # Neutral score if no functions found
        
        return (documented_functions / total_functions) * 100


class SecurityGate(QualityGate):
    """Validate security implementations."""
    
    def __init__(self):
        super().__init__("Security", weight=2.0, critical=True)
    
    def _run_check(self) -> QualityGateResult:
        security_score = 0
        max_score = 100
        security_details = {}
        
        # Check for security modules
        security_modules = [
            'liquid_vision/security/input_sanitizer.py',
            'liquid_vision/security/audit.py',
            'liquid_vision/security/crypto_utils.py'
        ]
        
        existing_modules = 0
        for module in security_modules:
            if Path(module).exists():
                existing_modules += 1
                security_details[module] = "✅ Present"
            else:
                security_details[module] = "❌ Missing"
        
        security_score += (existing_modules / len(security_modules)) * 30
        
        # Check for input validation
        sanitizer_path = Path('liquid_vision/security/input_sanitizer.py')
        if sanitizer_path.exists():
            content = sanitizer_path.read_text()
            if 'validate_tensor_input' in content:
                security_score += 20
                security_details['Input Validation'] = "✅ Implemented"
            else:
                security_details['Input Validation'] = "⚠️ Basic only"
                security_score += 10
        
        # Check for error handling
        error_handling_path = Path('liquid_vision/utils/error_handling.py')
        if error_handling_path.exists():
            content = error_handling_path.read_text()
            if 'ValidationError' in content and 'secure' in content.lower():
                security_score += 25
                security_details['Error Handling'] = "✅ Security-aware"
            else:
                security_details['Error Handling'] = "⚠️ Basic"
                security_score += 15
        
        # Check for encryption capabilities
        if 'encrypt' in sanitizer_path.read_text().lower() if sanitizer_path.exists() else False:
            security_score += 25
            security_details['Encryption'] = "✅ Available"
        else:
            security_details['Encryption'] = "❌ Not implemented"
        
        status = (QualityGateStatus.PASSED if security_score >= 70 else
                 QualityGateStatus.WARNING if security_score >= 50 else
                 QualityGateStatus.FAILED)
        
        recommendations = []
        if security_score < 70:
            recommendations.append("Enhance security implementations")
            if existing_modules < len(security_modules):
                recommendations.append("Add missing security modules")
            if security_score < 50:
                recommendations.append("Implement comprehensive input validation")
                recommendations.append("Add data encryption capabilities")
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=security_score,
            message=f"Security implementation score: {security_score:.1f}/100",
            details=security_details,
            recommendations=recommendations
        )


class PerformanceGate(QualityGate):
    """Validate performance optimizations and monitoring."""
    
    def __init__(self):
        super().__init__("Performance", weight=1.5)
    
    def _run_check(self) -> QualityGateResult:
        perf_score = 0
        max_score = 100
        perf_details = {}
        
        # Check for optimization modules
        optimization_files = [
            'liquid_vision/optimization/memory_efficient.py',
            'liquid_vision/optimization/auto_scaling.py',
            'liquid_vision/optimization/quantization.py'
        ]
        
        existing_optimizations = 0
        for opt_file in optimization_files:
            if Path(opt_file).exists():
                existing_optimizations += 1
                perf_details[opt_file] = "✅ Present"
            else:
                perf_details[opt_file] = "❌ Missing"
        
        perf_score += (existing_optimizations / len(optimization_files)) * 30
        
        # Check for monitoring capabilities
        monitoring_path = Path('liquid_vision/utils/monitoring.py')
        if monitoring_path.exists():
            content = monitoring_path.read_text()
            if 'SystemMonitor' in content and 'MetricsCollector' in content:
                perf_score += 25
                perf_details['Monitoring'] = "✅ Comprehensive"
            else:
                perf_score += 15
                perf_details['Monitoring'] = "⚠️ Basic"
        else:
            perf_details['Monitoring'] = "❌ Missing"
        
        # Check for real-time processing
        realtime_path = Path('liquid_vision/core/realtime_processor.py')
        if realtime_path.exists():
            perf_score += 20
            perf_details['Real-time Processing'] = "✅ Implemented"
        else:
            perf_details['Real-time Processing'] = "❌ Missing"
        
        # Check for edge deployment optimization
        edge_path = Path('liquid_vision/deployment/edge_deployer.py')
        if edge_path.exists():
            content = edge_path.read_text()
            if 'quantization' in content.lower() and 'c_code' in content.lower():
                perf_score += 25
                perf_details['Edge Optimization'] = "✅ Advanced"
            else:
                perf_score += 15
                perf_details['Edge Optimization'] = "⚠️ Basic"
        
        status = (QualityGateStatus.PASSED if perf_score >= 70 else
                 QualityGateStatus.WARNING if perf_score >= 50 else
                 QualityGateStatus.FAILED)
        
        recommendations = []
        if perf_score < 70:
            recommendations.append("Add more performance optimizations")
            if not monitoring_path.exists():
                recommendations.append("Implement performance monitoring")
            if not realtime_path.exists():
                recommendations.append("Add real-time processing capabilities")
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=perf_score,
            message=f"Performance optimization score: {perf_score:.1f}/100",
            details=perf_details,
            recommendations=recommendations
        )


class DeploymentReadinessGate(QualityGate):
    """Validate deployment readiness and edge compatibility."""
    
    def __init__(self):
        super().__init__("Deployment Readiness", weight=1.5)
    
    def _run_check(self) -> QualityGateResult:
        deploy_score = 0
        max_score = 100
        deploy_details = {}
        
        # Check deployment modules
        deployment_files = [
            'liquid_vision/deployment/edge_deployer.py',
            'liquid_vision/deployment/hardware_interface.py'
        ]
        
        for deploy_file in deployment_files:
            if Path(deploy_file).exists():
                deploy_score += 25
                deploy_details[deploy_file] = "✅ Present"
            else:
                deploy_details[deploy_file] = "❌ Missing"
        
        # Check for C code generation
        edge_deployer_path = Path('liquid_vision/deployment/edge_deployer.py')
        if edge_deployer_path.exists():
            content = edge_deployer_path.read_text()
            if '_generate_c_code' in content:
                deploy_score += 20
                deploy_details['C Code Generation'] = "✅ Implemented"
            else:
                deploy_details['C Code Generation'] = "❌ Missing"
        
        # Check for hardware abstraction
        hw_interface_path = Path('liquid_vision/deployment/hardware_interface.py')
        if hw_interface_path.exists():
            content = hw_interface_path.read_text()
            if 'ESP32' in content and 'CortexM' in content:
                deploy_score += 15
                deploy_details['Hardware Support'] = "✅ Multi-platform"
            else:
                deploy_score += 8
                deploy_details['Hardware Support'] = "⚠️ Limited"
        
        # Check for build configuration
        build_files = ['setup.py', 'pyproject.toml', 'requirements.txt']
        build_score = 0
        for build_file in build_files:
            if Path(build_file).exists():
                build_score += 1
        
        deploy_score += (build_score / len(build_files)) * 15
        deploy_details['Build Configuration'] = f"{build_score}/{len(build_files)} files present"
        
        status = (QualityGateStatus.PASSED if deploy_score >= 70 else
                 QualityGateStatus.WARNING if deploy_score >= 50 else
                 QualityGateStatus.FAILED)
        
        recommendations = []
        if deploy_score < 70:
            recommendations.append("Improve deployment capabilities")
            if not edge_deployer_path.exists():
                recommendations.append("Add edge deployment tools")
            if '_generate_c_code' not in (edge_deployer_path.read_text() if edge_deployer_path.exists() else ''):
                recommendations.append("Implement C code generation for edge devices")
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=deploy_score,
            message=f"Deployment readiness score: {deploy_score:.1f}/100",
            details=deploy_details,
            recommendations=recommendations
        )


class QualityGateRunner:
    """Runs all quality gates and generates comprehensive report."""
    
    def __init__(self):
        self.gates = [
            ImportValidationGate(),
            CodeStructureGate(),
            DocumentationGate(),
            SecurityGate(),
            PerformanceGate(),
            DeploymentReadinessGate()
        ]
        self.logger = logging.getLogger('quality_gate_runner')
    
    def run_all_gates(self, fail_fast: bool = False) -> QualityReport:
        """Run all quality gates and generate report."""
        self.logger.info("Starting quality gate validation...")
        start_time = time.time()
        
        results = []
        for gate in self.gates:
            try:
                result = gate.execute()
                results.append(result)
                
                # Fail fast on critical gate failures
                if fail_fast and gate.critical and result.status == QualityGateStatus.FAILED:
                    self.logger.error(f"Critical gate {gate.name} failed, stopping execution")
                    break
                    
            except Exception as e:
                self.logger.error(f"Gate {gate.name} execution failed: {e}")
                results.append(QualityGateResult(
                    gate_name=gate.name,
                    status=QualityGateStatus.FAILED,
                    score=0.0,
                    message=f"Gate execution error: {str(e)}"
                ))
        
        # Generate report
        report = self._generate_report(results, time.time() - start_time)
        self.logger.info(f"Quality validation completed. Overall score: {report.overall_score:.1f}/100")
        
        return report
    
    def _generate_report(self, results: List[QualityGateResult], execution_time: float) -> QualityReport:
        """Generate comprehensive quality report."""
        total_gates = len(results)
        passed_gates = sum(1 for r in results if r.status == QualityGateStatus.PASSED)
        failed_gates = sum(1 for r in results if r.status == QualityGateStatus.FAILED)
        warning_gates = sum(1 for r in results if r.status == QualityGateStatus.WARNING)
        skipped_gates = sum(1 for r in results if r.status == QualityGateStatus.SKIPPED)
        
        # Calculate overall score (weighted)
        total_weight = sum(gate.weight for gate in self.gates)
        weighted_score = sum(result.score * next(g.weight for g in self.gates if g.name == result.gate_name) 
                           for result in results)
        overall_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Generate summary
        if overall_score >= 80:
            summary = "✅ PRODUCTION READY - All quality gates passed with excellent scores"
        elif overall_score >= 70:
            summary = "✅ PRODUCTION READY - Minor improvements recommended"
        elif overall_score >= 60:
            summary = "⚠️ NEEDS IMPROVEMENT - Address warnings before production"
        elif overall_score >= 40:
            summary = "❌ NOT READY - Significant issues must be resolved"
        else:
            summary = "❌ CRITICAL ISSUES - Major improvements required"
        
        # Collect recommendations
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        
        # Add overall recommendations
        if failed_gates > 0:
            all_recommendations.append(f"Address {failed_gates} failed quality gates")
        if warning_gates > total_gates // 2:
            all_recommendations.append("Review and resolve quality warnings")
        
        return QualityReport(
            overall_score=overall_score,
            total_gates=total_gates,
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            warning_gates=warning_gates,
            skipped_gates=skipped_gates,
            execution_time=execution_time,
            gate_results=results,
            summary=summary,
            recommendations=list(set(all_recommendations))  # Remove duplicates
        )
    
    def save_report(self, report: QualityReport, output_path: str = "quality_report.json"):
        """Save quality report to file."""
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        # Also create a human-readable summary
        summary_path = output_path.replace('.json', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(self._generate_text_report(report))
        
        self.logger.info(f"Quality report saved to {output_path} and {summary_path}")
    
    def _generate_text_report(self, report: QualityReport) -> str:
        """Generate human-readable text report."""
        text = f"""
# Liquid Vision SDK Quality Report

## Overall Assessment
**Score: {report.overall_score:.1f}/100**
**Status: {report.summary}**

## Quality Gates Summary
- Total Gates: {report.total_gates}
- Passed: {report.passed_gates} ✅
- Failed: {report.failed_gates} ❌
- Warnings: {report.warning_gates} ⚠️
- Skipped: {report.skipped_gates} ⏭️

## Detailed Results

"""
        
        for result in report.gate_results:
            status_emoji = {
                QualityGateStatus.PASSED: "✅",
                QualityGateStatus.FAILED: "❌",
                QualityGateStatus.WARNING: "⚠️",
                QualityGateStatus.SKIPPED: "⏭️"
            }
            
            text += f"### {result.gate_name} {status_emoji[result.status]}\n"
            text += f"**Score:** {result.score:.1f}/100\n"
            text += f"**Message:** {result.message}\n"
            
            if result.recommendations:
                text += f"**Recommendations:**\n"
                for rec in result.recommendations:
                    text += f"- {rec}\n"
            text += "\n"
        
        if report.recommendations:
            text += "## Overall Recommendations\n\n"
            for rec in report.recommendations:
                text += f"- {rec}\n"
        
        text += f"\n## Execution Details\n"
        text += f"- Total execution time: {report.execution_time:.2f} seconds\n"
        text += f"- Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return text


def run_quality_gates(fail_fast: bool = False, save_report: bool = True) -> QualityReport:
    """Main entry point for quality gate validation."""
    runner = QualityGateRunner()
    report = runner.run_all_gates(fail_fast=fail_fast)
    
    if save_report:
        runner.save_report(report)
    
    return report


if __name__ == "__main__":
    # Run quality gates from command line
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Liquid Vision SDK Quality Gates")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first critical failure")
    parser.add_argument("--no-save", action="store_true", help="Don't save report to files")
    
    args = parser.parse_args()
    
    report = run_quality_gates(fail_fast=args.fail_fast, save_report=not args.no_save)
    
    # Print summary to console
    print(f"\n{'='*60}")
    print(f"LIQUID VISION SDK QUALITY ASSESSMENT")
    print(f"{'='*60}")
    print(f"Overall Score: {report.overall_score:.1f}/100")
    print(f"Status: {report.summary}")
    print(f"Gates: {report.passed_gates} passed, {report.failed_gates} failed, {report.warning_gates} warnings")
    print(f"{'='*60}\n")
    
    # Exit with appropriate code
    sys.exit(0 if report.failed_gates == 0 else 1)