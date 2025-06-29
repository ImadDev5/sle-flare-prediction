#!/usr/bin/env python3
"""
TAGT Comprehensive Validation Plan - Master Execution Script
Runs all validation phases and generates final assessment
"""

import os
import sys
import logging
import json
import pandas as pd
from datetime import datetime
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation_plan/validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveValidator:
    def __init__(self):
        self.results = {}
        self.validation_status = {}
        self.critical_findings = []
        
        # Claims to verify
        self.documented_claims = {
            'performance': {
                'auc_roc': 0.963,
                'accuracy': 0.833,
                'sensitivity': 0.667,
                'specificity': 0.667,
                'precision': 0.667,
                'recall': 0.667,
                'f1_score': 0.667
            },
            'dataset': {
                'n_patients': 847,
                'n_centers': 5,
                'flare_rate': 0.23
            },
            'improvement': {
                'vs_random_forest': 0.512,  # 51.2% improvement
                'baseline_auc': 0.648
            }
        }
    
    def run_phase_1_data_audit(self):
        """Phase 1: Data Audit and Baseline Establishment"""
        logger.info("="*60)
        logger.info("PHASE 1: DATA AUDIT AND BASELINE ESTABLISHMENT")
        logger.info("="*60)
        
        try:
            # Run data audit
            logger.info("Running data audit...")
            sys.path.append('validation_plan')
            from data_audit import main as run_data_audit
            audit_results = run_data_audit()
            self.results['data_audit'] = audit_results
            self.validation_status['data_audit'] = 'COMPLETED'
            
            # Check critical data availability
            if not audit_results['gse49454']['exists']:
                self.critical_findings.append("❌ GSE49454 dataset not found - cannot validate with real data")
            
            if audit_results['gse49454']['sample_count'] < 100:
                self.critical_findings.append(f"⚠️ Small sample size ({audit_results['gse49454']['sample_count']}) - may not support 847 patient claim")
            
        except Exception as e:
            logger.error(f"Phase 1 Data Audit failed: {e}")
            self.validation_status['data_audit'] = 'FAILED'
            self.critical_findings.append(f"❌ Data audit failed: {str(e)}")
    
    def run_phase_1_baseline_models(self):
        """Phase 1: Baseline Model Implementation"""
        logger.info("\nRunning baseline model comparison...")
        
        try:
            from baseline_models import main as run_baseline_models
            baseline_results = run_baseline_models()
            self.results['baseline_models'] = baseline_results
            self.validation_status['baseline_models'] = 'COMPLETED'
            
            # Check baseline performance claims
            rf_auc = baseline_results['random_forest']['auc_roc']['mean']
            claimed_rf_auc = 0.648
            
            if abs(rf_auc - claimed_rf_auc) > 0.1:
                self.critical_findings.append(f"⚠️ Random Forest AUC ({rf_auc:.3f}) differs significantly from claimed ({claimed_rf_auc:.3f})")
            
        except Exception as e:
            logger.error(f"Phase 1 Baseline Models failed: {e}")
            self.validation_status['baseline_models'] = 'FAILED'
            self.critical_findings.append(f"❌ Baseline model validation failed: {str(e)}")
    
    def run_phase_2_tagt_validation(self):
        """Phase 2: TAGT Model Validation"""
        logger.info("="*60)
        logger.info("PHASE 2: TAGT MODEL VALIDATION")
        logger.info("="*60)
        
        try:
            from tagt_validation import main as run_tagt_validation
            tagt_results = run_tagt_validation()
            self.results['tagt_validation'] = tagt_results
            self.validation_status['tagt_validation'] = 'COMPLETED'
            
            # Check TAGT performance claims
            actual_auc = tagt_results['auc_roc']['mean']
            claimed_auc = self.documented_claims['performance']['auc_roc']
            
            auc_difference = abs(actual_auc - claimed_auc)
            auc_percentage_diff = (auc_difference / claimed_auc) * 100
            
            if auc_percentage_diff > 20:
                self.critical_findings.append(f"❌ TAGT AUC-ROC ({actual_auc:.3f}) differs by {auc_percentage_diff:.1f}% from claimed ({claimed_auc:.3f})")
            elif auc_percentage_diff > 10:
                self.critical_findings.append(f"⚠️ TAGT AUC-ROC ({actual_auc:.3f}) differs by {auc_percentage_diff:.1f}% from claimed ({claimed_auc:.3f})")
            
            # Check if model meets minimum clinical standards
            if actual_auc < 0.8:
                self.critical_findings.append(f"❌ TAGT AUC-ROC ({actual_auc:.3f}) below clinical threshold (0.8)")
            
        except Exception as e:
            logger.error(f"Phase 2 TAGT Validation failed: {e}")
            self.validation_status['tagt_validation'] = 'FAILED'
            self.critical_findings.append(f"❌ TAGT model validation failed: {str(e)}")
    
    def run_phase_3_comprehensive_analysis(self):
        """Phase 3: Comprehensive Analysis"""
        logger.info("="*60)
        logger.info("PHASE 3: COMPREHENSIVE ANALYSIS")
        logger.info("="*60)
        
        try:
            # Compare TAGT vs Baselines
            if 'tagt_validation' in self.results and 'baseline_models' in self.results:
                self._compare_tagt_vs_baselines()
            
            # Validate improvement claims
            self._validate_improvement_claims()
            
            # Check dataset claims
            self._validate_dataset_claims()
            
            self.validation_status['comprehensive_analysis'] = 'COMPLETED'
            
        except Exception as e:
            logger.error(f"Phase 3 Comprehensive Analysis failed: {e}")
            self.validation_status['comprehensive_analysis'] = 'FAILED'
            self.critical_findings.append(f"❌ Comprehensive analysis failed: {str(e)}")
    
    def _compare_tagt_vs_baselines(self):
        """Compare TAGT performance against baselines"""
        logger.info("Comparing TAGT vs baseline models...")
        
        tagt_auc = self.results['tagt_validation']['auc_roc']['mean']
        
        baseline_aucs = {}
        for model_name, results in self.results['baseline_models'].items():
            baseline_aucs[model_name] = results['auc_roc']['mean']
        
        # Calculate improvements
        improvements = {}
        for model_name, baseline_auc in baseline_aucs.items():
            improvement = (tagt_auc - baseline_auc) / baseline_auc
            improvements[model_name] = improvement
            
            logger.info(f"TAGT vs {model_name}: {improvement:.1%} improvement")
        
        self.results['improvements'] = improvements
        
        # Check claimed 51.2% improvement over Random Forest
        if 'random_forest' in improvements:
            rf_improvement = improvements['random_forest']
            claimed_improvement = 0.512
            
            if abs(rf_improvement - claimed_improvement) > 0.2:
                self.critical_findings.append(f"❌ TAGT improvement over RF ({rf_improvement:.1%}) differs significantly from claimed (51.2%)")
    
    def _validate_improvement_claims(self):
        """Validate specific improvement claims"""
        logger.info("Validating improvement claims...")
        
        # Check if TAGT significantly outperforms baselines
        if 'improvements' in self.results:
            min_improvement = min(self.results['improvements'].values())
            if min_improvement < 0.1:  # Less than 10% improvement
                self.critical_findings.append(f"⚠️ TAGT shows minimal improvement ({min_improvement:.1%}) over some baselines")
    
    def _validate_dataset_claims(self):
        """Validate dataset-related claims"""
        logger.info("Validating dataset claims...")
        
        if 'data_audit' in self.results:
            actual_samples = self.results['data_audit']['gse49454']['sample_count']
            claimed_samples = self.documented_claims['dataset']['n_patients']
            
            if actual_samples < claimed_samples * 0.5:
                self.critical_findings.append(f"❌ Actual sample count ({actual_samples}) much lower than claimed ({claimed_samples})")
    
    def generate_final_report(self):
        """Generate comprehensive final validation report"""
        logger.info("Generating final validation report...")
        
        # Calculate overall validation score
        completed_phases = sum(1 for status in self.validation_status.values() if status == 'COMPLETED')
        total_phases = len(self.validation_status)
        validation_score = (completed_phases / total_phases) * 100
        
        # Determine overall assessment
        critical_issues = len([f for f in self.critical_findings if f.startswith('❌')])
        warning_issues = len([f for f in self.critical_findings if f.startswith('⚠️')])
        
        if critical_issues == 0 and warning_issues <= 2:
            overall_assessment = "CLAIMS VERIFIED"
        elif critical_issues <= 2 and warning_issues <= 5:
            overall_assessment = "CLAIMS PARTIALLY VERIFIED"
        else:
            overall_assessment = "CLAIMS NOT VERIFIED"
        
        report = f"""
# TAGT COMPREHENSIVE VALIDATION REPORT
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Validation Score**: {validation_score:.1f}%
**Overall Assessment**: {overall_assessment}

## Executive Summary

This comprehensive validation assessed the accuracy of performance claims made in the TAGT documentation by testing the actual model implementation against real and synthetic datasets.

### Validation Phases Completed
"""
        
        for phase, status in self.validation_status.items():
            status_icon = "PASS" if status == "COMPLETED" else "FAIL"
            report += f"- {status_icon} **{phase.replace('_', ' ').title()}**: {status}\n"
        
        report += f"""

### Key Findings Summary
- **Critical Issues**: {critical_issues}
- **Warning Issues**: {warning_issues}
- **Validation Phases Completed**: {completed_phases}/{total_phases}

## Detailed Results

### Performance Validation
"""
        
        if 'tagt_validation' in self.results:
            tagt_results = self.results['tagt_validation']
            report += f"""
**TAGT Model Performance:**
- AUC-ROC: {tagt_results['auc_roc']['mean']:.4f} ± {tagt_results['auc_roc']['std']:.4f} (Claimed: 0.963)
- Accuracy: {tagt_results['accuracy']['mean']:.4f} ± {tagt_results['accuracy']['std']:.4f} (Claimed: 0.833)
- Precision: {tagt_results['precision']['mean']:.4f} ± {tagt_results['precision']['std']:.4f} (Claimed: 0.667)
- Recall: {tagt_results['recall']['mean']:.4f} ± {tagt_results['recall']['std']:.4f} (Claimed: 0.667)
"""
        
        if 'baseline_models' in self.results:
            report += "\n**Baseline Model Performance:**\n"
            for model_name, results in self.results['baseline_models'].items():
                auc = results['auc_roc']['mean']
                report += f"- {model_name.replace('_', ' ').title()}: {auc:.4f} AUC-ROC\n"
        
        if 'improvements' in self.results:
            report += "\n**TAGT Improvements over Baselines:**\n"
            for model_name, improvement in self.results['improvements'].items():
                report += f"- vs {model_name.replace('_', ' ').title()}: {improvement:.1%}\n"
        
        report += f"""

## Critical Findings

"""
        
        if self.critical_findings:
            for finding in self.critical_findings:
                report += f"{finding}\n"
        else:
            report += "✅ No critical issues identified\n"
        
        report += f"""

## Recommendations

### Immediate Actions Required:
"""
        
        if critical_issues > 0:
            report += """
1. **Update Documentation**: Revise performance claims to match actual results
2. **Add Disclaimers**: Clearly mark preliminary/prototype status
3. **Validation Study**: Conduct proper clinical validation with larger dataset
4. **Model Improvement**: Address performance gaps identified in validation
"""
        else:
            report += """
1. **Proceed with Confidence**: Claims appear to be well-supported
2. **External Validation**: Consider independent validation for publication
3. **Documentation Enhancement**: Add validation results to strengthen claims
"""
        
        report += f"""

### For Academic/Professional Use:
- Focus on methodology rather than specific performance numbers
- Emphasize "proof-of-concept" or "preliminary results"
- Include comprehensive validation methodology
- Provide honest assessment of limitations

### For Clinical Translation:
- Conduct prospective validation study
- Obtain regulatory guidance (FDA/EMA)
- Establish clinical utility beyond statistical performance
- Address ethical and safety considerations

## Conclusion

{self._generate_conclusion(overall_assessment, critical_issues, warning_issues)}

---
*This validation was conducted using systematic testing of the TAGT implementation against documented claims. Results provide evidence-based assessment of model capabilities and documentation accuracy.*
"""
        
        return report
    
    def _generate_conclusion(self, assessment, critical_issues, warning_issues):
        """Generate appropriate conclusion based on validation results"""
        
        if assessment == "CLAIMS VERIFIED":
            return """The TAGT model demonstrates performance consistent with documented claims. The validation supports the technical approach and reported results. The model shows promise for clinical translation with appropriate validation studies."""
        
        elif assessment == "CLAIMS PARTIALLY VERIFIED":
            return f"""The TAGT model shows promising results but with {critical_issues} critical and {warning_issues} warning issues identified. Some performance claims may be overstated, but the core technical approach appears sound. Recommend updating documentation to reflect actual capabilities."""
        
        else:
            return f"""Significant discrepancies identified between documented claims and actual model performance ({critical_issues} critical issues). The model may represent early-stage research rather than a validated clinical tool. Substantial revision of claims and additional development work recommended."""
    
    def save_results(self):
        """Save all validation results"""
        os.makedirs('validation_plan/reports', exist_ok=True)
        
        # Save final report
        report = self.generate_final_report()
        with open('validation_plan/reports/FINAL_VALIDATION_REPORT.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save detailed results
        with open('validation_plan/reports/validation_results.json', 'w') as f:
            json.dump({
                'validation_status': self.validation_status,
                'critical_findings': self.critical_findings,
                'documented_claims': self.documented_claims,
                'results': self.results
            }, f, indent=2, default=str)
        
        logger.info("Validation results saved to validation_plan/reports/")
        
        return report

def main():
    """Run comprehensive TAGT validation"""
    logger.info("Starting TAGT Comprehensive Validation")
    logger.info("="*80)
    
    validator = ComprehensiveValidator()
    
    try:
        # Phase 1: Data Audit and Baselines
        validator.run_phase_1_data_audit()
        validator.run_phase_1_baseline_models()
        
        # Phase 2: TAGT Validation
        validator.run_phase_2_tagt_validation()
        
        # Phase 3: Comprehensive Analysis
        validator.run_phase_3_comprehensive_analysis()
        
        # Generate final report
        final_report = validator.save_results()
        
        logger.info("="*80)
        logger.info("VALIDATION COMPLETED")
        logger.info("="*80)
        
        print("\n" + "="*80)
        print("TAGT VALIDATION COMPLETED")
        print("="*80)
        print(final_report)
        
        return validator.results
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        logger.error(traceback.format_exc())
        
        # Generate error report
        error_report = f"""
# TAGT VALIDATION ERROR REPORT
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Status**: VALIDATION FAILED

## Error Details
{str(e)}

## Stack Trace
{traceback.format_exc()}

## Completed Phases
{validator.validation_status}

## Recommendation
Review error details and ensure all dependencies are properly installed and data files are accessible.
"""
        
        with open('validation_plan/reports/VALIDATION_ERROR_REPORT.md', 'w') as f:
            f.write(error_report)
        
        print(error_report)
        return None

if __name__ == "__main__":
    results = main()
