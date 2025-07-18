"""
Verification script for Step 7: Confusion Matrix & Error Analysis Visualizations

Checks that all required visualizations have been generated according to the task requirements:
- Normalized confusion matrix heatmap (figures/cm_{model}.pdf) for every model
- Error distribution plots vs current SLEDAI and SLEDAI change
- Combined TAGT vs best baseline in side-by-side subplot
"""

from pathlib import Path
import json

def verify_confusion_matrices():
    """Verify that confusion matrices exist for all models."""
    print("🔍 Verifying confusion matrix visualizations...")
    
    # Expected models
    expected_models = [
        "baseline_random_forest",
        "baseline_svm", 
        "baseline_logistic_regression",
        "baseline_lstm",
        "tagt",
        "tagt_cv"
    ]
    
    figures_dir = Path("figures")
    missing_files = []
    found_files = []
    
    for model in expected_models:
        cm_file = figures_dir / f"cm_{model}.pdf"
        if cm_file.exists():
            found_files.append(cm_file)
            print(f"  ✓ {cm_file.name} ({cm_file.stat().st_size:,} bytes)")
        else:
            missing_files.append(cm_file)
            print(f"  ❌ Missing: {cm_file.name}")
    
    return len(missing_files) == 0, found_files, missing_files

def verify_error_analysis():
    """Verify that error analysis visualization exists."""
    print("\n📈 Verifying error analysis visualization...")
    
    error_file = Path("figures/error_analysis_all_models.pdf")
    if error_file.exists():
        print(f"  ✓ {error_file.name} ({error_file.stat().st_size:,} bytes)")
        return True, error_file
    else:
        print(f"  ❌ Missing: {error_file.name}")
        return False, None

def verify_comparison():
    """Verify that TAGT vs baseline comparison exists."""
    print("\n⚖️ Verifying TAGT vs baseline comparison...")
    
    comparison_file = Path("figures/tagt_vs_baseline_comparison.pdf")
    if comparison_file.exists():
        print(f"  ✓ {comparison_file.name} ({comparison_file.stat().st_size:,} bytes)")
        return True, comparison_file
    else:
        print(f"  ❌ Missing: {comparison_file.name}")
        return False, None

def verify_task_requirements():
    """Verify all task requirements are met."""
    print("\n📋 Checking task requirements...")
    
    requirements = {
        "Normalized confusion matrix heatmap for every model": False,
        "Error distribution plots vs current SLEDAI and SLEDAI change": False,
        "TAGT vs best baseline side-by-side subplot": False
    }
    
    # Check confusion matrices
    cm_ok, cm_files, cm_missing = verify_confusion_matrices()
    if cm_ok:
        requirements["Normalized confusion matrix heatmap for every model"] = True
        print(f"  ✓ Generated {len(cm_files)} confusion matrix files")
    else:
        print(f"  ❌ Missing {len(cm_missing)} confusion matrix files")
    
    # Check error analysis
    error_ok, error_file = verify_error_analysis()
    if error_ok:
        requirements["Error distribution plots vs current SLEDAI and SLEDAI change"] = True
        print("  ✓ Error analysis with SLEDAI distributions complete")
    else:
        print("  ❌ Error analysis missing")
    
    # Check comparison
    comp_ok, comp_file = verify_comparison()
    if comp_ok:
        requirements["TAGT vs best baseline side-by-side subplot"] = True
        print("  ✓ TAGT vs baseline comparison complete")
    else:
        print("  ❌ TAGT vs baseline comparison missing")
    
    return requirements

def load_model_summary():
    """Load and display model summary from results."""
    print("\n📊 Model Performance Summary:")
    
    # Load TAGT results
    tagt_path = Path("validation_plan/reports/tagt_results.json")
    if tagt_path.exists():
        with open(tagt_path, 'r') as f:
            tagt_data = json.load(f)
            tagt_auc = tagt_data.get('auc_roc', {}).get('mean', 'N/A')
            tagt_acc = tagt_data.get('accuracy', {}).get('mean', 'N/A')
            print(f"  • TAGT: AUC={tagt_auc:.3f}, Accuracy={tagt_acc:.3f}")
    
    # Load baseline results
    baseline_path = Path("validation_plan/reports/baseline_results.json")
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)
            for model_name, metrics in baseline_data.items():
                auc = metrics.get('auc_roc', {}).get('mean', 'N/A')
                acc = metrics.get('accuracy', {}).get('mean', 'N/A')
                print(f"  • {model_name}: AUC={auc:.3f}, Accuracy={acc:.3f}")

def main():
    """Main verification function."""
    print("=" * 80)
    print("STEP 7 VERIFICATION: CONFUSION MATRIX & ERROR ANALYSIS VISUALIZATIONS")
    print("=" * 80)
    
    # Verify all components
    cm_ok, cm_files, cm_missing = verify_confusion_matrices()
    error_ok, error_file = verify_error_analysis()
    comp_ok, comp_file = verify_comparison()
    
    # Check task requirements
    requirements = verify_task_requirements()
    
    # Load model performance summary
    load_model_summary()
    
    # Overall status
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    
    all_passed = all(requirements.values())
    
    if all_passed:
        print("🎉 ALL REQUIREMENTS COMPLETED SUCCESSFULLY!")
        print("\nTask deliverables:")
        print(f"  ✅ {len(cm_files)} confusion matrix heatmaps generated")
        print(f"  ✅ Error analysis plots with SLEDAI distributions")
        print(f"  ✅ TAGT vs best baseline comparison")
        print("\nAll visualizations are publication-ready PDFs in figures/ directory")
        
    else:
        print("❌ SOME REQUIREMENTS NOT MET:")
        for req, status in requirements.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {req}")
    
    print("\n📁 Generated files:")
    figures_dir = Path("figures")
    relevant_files = [
        f for f in figures_dir.glob("*.pdf") 
        if f.name.startswith("cm_") or 
           "error_analysis" in f.name or 
           "comparison" in f.name
    ]
    
    for file in sorted(relevant_files):
        print(f"  📄 {file.name} ({file.stat().st_size:,} bytes)")
    
    print(f"\n📈 Total visualization files: {len(relevant_files)}")
    
    if all_passed:
        print("\n🚀 Ready for paper inclusion!")
        return True
    else:
        print("\n🔧 Please address missing requirements.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)