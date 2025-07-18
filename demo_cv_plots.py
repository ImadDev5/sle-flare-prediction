Demonstrate statistical significance testing."""
    print("\n🔬 DEMO 2: Statistical Significance Testing")
    print("-" * 50)
    
    plotter = CVPerformancePlotter()
    df = plotter.prepare_plot_data()
    
    # Compute significance matrix for AUC
    print("Computing significance matrix for AUC...")
    sig_matrix = plotter.compute_significance_matrix(df, 'AUC')
    
    print("\n📋 Significance Matrix (p-values):")
    print(sig_matrix.round(4))
    
    # Identify significant comparisons
    print(f"\n⭐ Significant Comparisons (p < 0.05):")
    models = sig_matrix.index
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i < j:  # Only upper triangle
                p_val = sig_matrix.loc[model1, model2]
                if p_val < 0.05:
                    stars = plotter.get_significance_stars(p_val)
                    print(f"   • {model1} vs {model2}: p = {p_val:.4f} {stars}")
    
    return sig_matrix

def demo_single_plot_creation():
    """Demonstrate creating individual plots."""
    print("\n🎨 DEMO 3: Creating Individual Plots")
    print("-" * 50)
    
    plotter = CVPerformancePlotter()
    df = plotter.prepare_plot_data()
    
        print("Creating combined boxplot for AUC and Accuracy...")
    fig = plotter.create_combined_boxplot(df, ['AUC', 'Accuracy'])
    
    # Show plot (in a real environment, this would display the plot)
    print("✅ Plot created successfully!")
    print("   • Boxplots show quartiles and median")
    print("   • Strip plots show individual fold values") 
    print("   • Significance stars compare against best model")
    print("   • Sample sizes shown below each box")
    
    plt.close(fig)  # Close to prevent display in headless environment
    
    return fig

def demo_detailed_comparison():
    """Demonstrate detailed model comparison."""
    print("\n🔍 DEMO 4: Detailed Model Comparison")
    print("-" * 50)
    
    plotter = CVPerformancePlotter()
    df = plotter.prepare_plot_data()
    
    # Find best two models by AUC
    model_aucs = df.groupby('Model')['AUC'].mean().sort_values(ascending=False)
    best_model = model_aucs.index[0]
    second_model = model_aucs.index[1]
    
    print(f"Comparing top 2 models: {best_model} vs {second_model}")
    print(f"   • {best_model}: AUC = {model_aucs[best_model]:.3f}")
    print(f"   • {second_model}: AUC = {model_aucs[second_model]:.3f}")
    
        fig = plotter.create_detailed_comparison_plot(df, best_model, second_model)
    
    print("✅ Detailed comparison created!")
    print("   • Statistical test performed")
    print("   • P-value and effect size displayed")
    print("   • Individual data points shown")
    
    plt.close(fig)
    
    return fig

def demo_heatmap_visualization():
    """Demonstrate significance heatmap creation."""
    print("\n🔥 DEMO 5: Significance Heatmap")
    print("-" * 50)
    
    plotter = CVPerformancePlotter()
    df = plotter.prepare_plot_data()
    
        print("Creating significance heatmap for AUC...")
    fig = plotter.create_performance_heatmap(df, 'AUC')
    
    print("✅ Heatmap created!")
    print("   • Color intensity shows -log10(p-value)")
    print("   • Annotations show significance stars")
    print("   • Symmetric matrix (model A vs B = B vs A)")
    print("   • Diagonal shows self-comparison (p=1.0)")
    
    plt.close(fig)
    
    return fig

def demo_step3_integration():
    """Demonstrate integration with step 3 significance results."""
    print("\n🔗 DEMO 6: Step 3 Significance Integration")
    print("-" * 50)
    
    integrator = Step3SignificanceIntegrator()
    
    # Try to load step 3 results
    step3_matrix = integrator.load_step3_significance_matrix('auc')
    
    if step3_matrix is not None:
        print("✅ Found step 3 significance matrix!")
        print(f"   • Shape: {step3_matrix.shape}")
        print(f"   • Models: {list(step3_matrix.index)}")
    else:
        print("ℹ️  No step 3 significance matrix found")
        print("   • Will use computed bootstrap significance instead")
        print("   • This is normal if step 3 hasn't been run yet")
    
        try:
        fig = integrator.create_integrated_boxplot_with_step3_pvalues('AUC')
        print("✅ Integrated plot created!")
        plt.close(fig)
    except Exception as e:
        print(f"⚠️  Error creating integrated plot: {e}")
    
    return step3_matrix

def demo_full_pipeline():
    """Demonstrate the full plotting pipeline."""
    print("\n🚀 DEMO 7: Full Plotting Pipeline")
    print("-" * 50)
    
    plotter = CVPerformancePlotter()
    
        print("Running full plotting pipeline...")
    saved_plots = plotter.generate_all_plots()
    
    print(f"\n✅ Generated {len(saved_plots)} plots:")
    for plot_name, plot_path in saved_plots.items():
        print(f"   • {plot_name}")
        print(f"     📁 {plot_path}")
    
        summary_path = plotter.save_summary_table()
    print(f"\n📊 Summary table: {summary_path}")
    
    return saved_plots

def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("CROSS-VALIDATION PERFORMANCE PLOTTING DEMO")
    print("=" * 60)
    print("This demo shows how to use the CV performance plotting tools")
    print("for creating publication-quality visualizations with statistical testing.")
    print()
    
    try:
        # Run all demos
        df = demo_basic_plotting()
        sig_matrix = demo_significance_testing()
        boxplot_fig = demo_single_plot_creation()
        comparison_fig = demo_detailed_comparison()
        heatmap_fig = demo_heatmap_visualization()
        step3_matrix = demo_step3_integration()
        saved_plots = demo_full_pipeline()
        
        # Final summary
        print("\n" + "=" * 60)
        print("DEMO COMPLETE! 🎉")
        print("=" * 60)
        print("Key features demonstrated:")
        print("✅ Per-fold performance visualization")
        print("✅ Statistical significance testing (bootstrap)")
        print("✅ Multiple plot types (boxplot, heatmap, comparisons)")
        print("✅ Step 3 significance integration")
        print("✅ Automated plot generation pipeline")
        print("✅ Publication-quality formatting")
        
        print(f"\n📁 All plots saved to: results/plots")
        print("\n🎯 Ready for publication! 📄")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()