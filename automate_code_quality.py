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