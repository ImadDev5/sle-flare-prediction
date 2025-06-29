# Advanced Code Quality Automation for SLE TAGT Project
# PowerShell Script with System Integration and Monitoring

param(
    [switch]$SkipInstall,
    [switch]$QuickScan,
    [switch]$FullReport,
    [switch]$MonitorMode,
    [string]$ConfigFile = "quality_config.json"
)

# Set execution policy for current session
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

# Color output functions
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success { param([string]$Message) Write-ColorOutput $Message "Green" }
function Write-Warning { param([string]$Message) Write-ColorOutput $Message "Yellow" }
function Write-Error { param([string]$Message) Write-ColorOutput $Message "Red" }
function Write-Info { param([string]$Message) Write-ColorOutput $Message "Cyan" }

# Header
Write-ColorOutput "" 
Write-ColorOutput "🚀 SLE TAGT Advanced Code Quality Automation" "Magenta"
Write-ColorOutput "=" * 60 "Magenta"
Write-ColorOutput ""

# System Information
function Get-SystemInfo {
    Write-Info "📊 Gathering System Information..."
    
    $systemInfo = @{
        OS = "$($env:OS) $(Get-WmiObject -Class Win32_OperatingSystem | Select-Object -ExpandProperty Caption)"
        Architecture = $env:PROCESSOR_ARCHITECTURE
        Cores = $env:NUMBER_OF_PROCESSORS
        PowerShellVersion = $PSVersionTable.PSVersion.ToString()
        DotNetVersion = [System.Runtime.InteropServices.RuntimeInformation]::FrameworkDescription
        Memory = [math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
        DiskSpace = [math]::Round((Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='C:'").FreeSpace / 1GB, 2)
    }
    
    Write-Success "System: $($systemInfo.OS)"
    Write-Success "Architecture: $($systemInfo.Architecture)"
    Write-Success "CPU Cores: $($systemInfo.Cores)"
    Write-Success "Memory: $($systemInfo.Memory) GB"
    Write-Success "Free Disk: $($systemInfo.DiskSpace) GB"
    Write-Success "PowerShell: $($systemInfo.PowerShellVersion)"
    
    return $systemInfo
}

# GPU Information
function Get-GPUInfo {
    Write-Info "🖥️ Checking GPU Information..."
    
    try {
        $gpu = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -notlike "*Basic*" } | Select-Object -First 1
        if ($gpu) {
            Write-Success "GPU: $($gpu.Name)"
            Write-Success "GPU Memory: $([math]::Round($gpu.AdapterRAM / 1GB, 2)) GB"
            
            # Check NVIDIA GPU specifically
            if ($gpu.Name -like "*NVIDIA*") {
                try {
                    $nvidiaSmi = nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits 2>$null
                    if ($nvidiaSmi) {
                        Write-Success "NVIDIA GPU detected with nvidia-smi support"
                        Write-Info $nvidiaSmi
                    }
                } catch {
                    Write-Warning "nvidia-smi not available"
                }
            }
            
            return @{ Available = $true; Name = $gpu.Name; Memory = $gpu.AdapterRAM }
        } else {
            Write-Warning "No dedicated GPU detected"
            return @{ Available = $false }
        }
    } catch {
        Write-Warning "Could not retrieve GPU information: $($_.Exception.Message)"
        return @{ Available = $false }
    }
}

# Python Environment Check
function Test-PythonEnvironment {
    Write-Info "🐍 Checking Python Environment..."
    
    # Check if virtual environment exists
    if (Test-Path "venv_gpu\Scripts\Activate.ps1") {
        Write-Success "Virtual environment found"
        & .\venv_gpu\Scripts\Activate.ps1
        Write-Success "Virtual environment activated"
    } elseif (Test-Path "venv\Scripts\Activate.ps1") {
        Write-Success "Alternative virtual environment found"
        & .\venv\Scripts\Activate.ps1
        Write-Success "Virtual environment activated"
    } else {
        Write-Warning "No virtual environment found, using system Python"
    }
    
    # Check Python availability
    try {
        $pythonVersion = python --version 2>&1
        Write-Success "Python: $pythonVersion"
        
        # Check key packages
        $packages = @("torch", "numpy", "pandas", "scikit-learn")
        foreach ($package in $packages) {
            try {
                $version = python -c "import $package; print($package.__version__)" 2>$null
                if ($version) {
                    Write-Success "$package: $version"
                } else {
                    Write-Warning "$package: Not installed"
                }
            } catch {
                Write-Warning "$package: Not available"
            }
        }
        
        return $true
    } catch {
        Write-Error "Python not found in PATH"
        return $false
    }
}

# Install Quality Tools
function Install-QualityTools {
    if ($SkipInstall) {
        Write-Info "Skipping tool installation (--SkipInstall flag)"
        return
    }
    
    Write-Info "🔧 Installing Code Quality Tools..."
    
    $tools = @(
        "black", "isort", "flake8", "mypy", "bandit", 
        "pytest", "pytest-cov", "safety", "vulture", 
        "radon", "autopep8", "pre-commit"
    )
    
    foreach ($tool in $tools) {
        Write-Info "Installing $tool..."
        try {
            $result = python -m pip install $tool --quiet 2>&1
            Write-Success "✅ $tool installed"
        } catch {
            Write-Warning "⚠️ Failed to install $tool"
        }
    }
}

# Run Quality Automation
function Start-QualityAutomation {
    Write-Info "🚀 Starting Code Quality Automation..."
    
    try {
        if ($QuickScan) {
            Write-Info "Running quick scan mode..."
            $result = python -c "
import sys
sys.path.append('.')
from automate_code_quality import CodeQualityAutomator
automator = CodeQualityAutomator()
results = {
    'formatting': automator.format_code(),
    'linting': automator.run_linting(),
    'security': automator.run_security_scan()
}
print('Quick scan completed')
"
        } else {
            Write-Info "Running full automation..."
            $result = python automate_code_quality.py
        }
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "✅ Automation completed successfully"
            return $true
        } else {
            Write-Error "❌ Automation failed with exit code $LASTEXITCODE"
            return $false
        }
    } catch {
        Write-Error "❌ Automation failed: $($_.Exception.Message)"
        return $false
    }
}

# Monitor Mode
function Start-MonitorMode {
    Write-Info "👁️ Starting Monitor Mode..."
    Write-Info "Watching for file changes in src/ directory..."
    Write-Info "Press Ctrl+C to stop monitoring"
    
    $watcher = New-Object System.IO.FileSystemWatcher
    $watcher.Path = "src"
    $watcher.Filter = "*.py"
    $watcher.IncludeSubdirectories = $true
    $watcher.EnableRaisingEvents = $true
    
    $action = {
        $path = $Event.SourceEventArgs.FullPath
        $changeType = $Event.SourceEventArgs.ChangeType
        $timeStamp = $Event.TimeGenerated
        
        Write-Info "[$timeStamp] File $changeType: $path"
        
        # Run quick quality check on changed file
        if ($changeType -eq "Changed" -and $path -like "*.py") {
            Write-Info "Running quick quality check..."
            try {
                python -m black --check $path 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "✅ $path formatting OK"
                } else {
                    Write-Warning "⚠️ $path needs formatting"
                }
                
                python -m flake8 $path 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "✅ $path linting OK"
                } else {
                    Write-Warning "⚠️ $path has linting issues"
                }
            } catch {
                Write-Warning "Could not check $path"
            }
        }
    }
    
    Register-ObjectEvent -InputObject $watcher -EventName "Changed" -Action $action
    Register-ObjectEvent -InputObject $watcher -EventName "Created" -Action $action
    Register-ObjectEvent -InputObject $watcher -EventName "Deleted" -Action $action
    
    try {
        while ($true) {
            Start-Sleep 1
        }
    } finally {
        $watcher.EnableRaisingEvents = $false
        $watcher.Dispose()
        Write-Info "Monitor mode stopped"
    }
}

# Generate Advanced Report
function New-AdvancedReport {
    Write-Info "📊 Generating Advanced Quality Report..."
    
    $reportPath = "quality_reports\advanced_report.html"
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    $systemInfo = Get-SystemInfo
    $gpuInfo = Get-GPUInfo
    
    # Check if reports exist
    $reports = @()
    $reportFiles = @(
        "flake8_report.txt",
        "mypy_report.txt", 
        "bandit_report.json",
        "safety_report.json",
        "complexity_report.json",
        "dead_code_report.txt"
    )
    
    foreach ($file in $reportFiles) {
        $fullPath = "quality_reports\$file"
        if (Test-Path $fullPath) {
            $reports += @{ Name = $file; Exists = $true; Size = (Get-Item $fullPath).Length }
        } else {
            $reports += @{ Name = $file; Exists = $false; Size = 0 }
        }
    }
    
    $html = @"
<!DOCTYPE html>
<html>
<head>
    <title>SLE TAGT - Advanced Quality Report</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; }
        .section { margin: 20px; padding: 20px; border-radius: 8px; }
        .system-info { background: #e8f4fd; border-left: 4px solid #2196F3; }
        .gpu-info { background: #f3e5f5; border-left: 4px solid #9C27B0; }
        .reports-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .report-card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 15px; }
        .status-ok { border-left: 4px solid #4CAF50; }
        .status-warning { border-left: 4px solid #FF9800; }
        .status-error { border-left: 4px solid #F44336; }
        .metric { display: inline-block; margin: 5px 10px; padding: 8px 12px; background: #f8f9fa; border-radius: 4px; }
        .footer { text-align: center; padding: 20px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 SLE TAGT Advanced Quality Report</h1>
            <p>Generated on $timestamp</p>
            <p>Comprehensive code quality analysis and system optimization report</p>
        </div>
        
        <div class="section system-info">
            <h2>🖥️ System Information</h2>
            <div class="metric">OS: $($systemInfo.OS)</div>
            <div class="metric">Architecture: $($systemInfo.Architecture)</div>
            <div class="metric">CPU Cores: $($systemInfo.Cores)</div>
            <div class="metric">Memory: $($systemInfo.Memory) GB</div>
            <div class="metric">Free Disk: $($systemInfo.DiskSpace) GB</div>
            <div class="metric">PowerShell: $($systemInfo.PowerShellVersion)</div>
        </div>
        
        <div class="section gpu-info">
            <h2>🎮 GPU Information</h2>
            $(if ($gpuInfo.Available) {
                "<div class='metric'>GPU: $($gpuInfo.Name)</div>"
                "<div class='metric'>Memory: $([math]::Round($gpuInfo.Memory / 1GB, 2)) GB</div>"
                "<div class='metric'>Status: ✅ Available</div>"
            } else {
                "<div class='metric'>Status: ❌ No dedicated GPU detected</div>"
            })
        </div>
        
        <div class="section">
            <h2>📊 Quality Reports Status</h2>
            <div class="reports-grid">
"@
    
    foreach ($report in $reports) {
        $statusClass = if ($report.Exists) { "status-ok" } else { "status-error" }
        $statusIcon = if ($report.Exists) { "✅" } else { "❌" }
        $sizeText = if ($report.Exists) { "$([math]::Round($report.Size / 1KB, 1)) KB" } else { "Not generated" }
        
        $html += @"
                <div class="report-card $statusClass">
                    <h3>$statusIcon $($report.Name)</h3>
                    <p>Size: $sizeText</p>
                    $(if ($report.Exists) { "<a href='$($report.Name)'>View Report</a>" } else { "<span style='color: #666;'>Report not available</span>" })
                </div>
"@
    }
    
    $html += @"
            </div>
        </div>
        
        <div class="section">
            <h2>🎯 Optimization Recommendations</h2>
            <ul>
                <li>🔄 Run automated formatting before each commit</li>
                <li>📈 Maintain test coverage above 80%</li>
                <li>🔍 Keep cyclomatic complexity below 10</li>
                <li>🛡️ Regular security dependency updates</li>
                $(if ($gpuInfo.Available) { "<li>⚡ Optimize GPU memory usage for training</li>" } else { "<li>💻 Consider GPU acceleration for training</li>" })
                <li>📚 Keep documentation up to date</li>
                <li>🧪 Add more unit tests for critical functions</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>Generated by SLE TAGT Advanced Quality Automation System</p>
            <p>For more information, check the individual report files</p>
        </div>
    </div>
</body>
</html>
"@
    
    # Ensure directory exists
    if (!(Test-Path "quality_reports")) {
        New-Item -ItemType Directory -Path "quality_reports" -Force | Out-Null
    }
    
    $html | Out-File -FilePath $reportPath -Encoding UTF8
    Write-Success "Advanced report generated: $reportPath"
    
    return $reportPath
}

# Main Execution
try {
    # Get system information
    $systemInfo = Get-SystemInfo
    $gpuInfo = Get-GPUInfo
    
    # Check Python environment
    if (!(Test-PythonEnvironment)) {
        Write-Error "Python environment check failed"
        exit 1
    }
    
    # Handle different modes
    if ($MonitorMode) {
        Start-MonitorMode
    } else {
        # Install tools
        Install-QualityTools
        
        # Run automation
        $success = Start-QualityAutomation
        
        if ($success -or $FullReport) {
            # Generate advanced report
            $reportPath = New-AdvancedReport
            
            Write-Success ""
            Write-Success "✅ Quality automation completed!"
            Write-Success "📊 Reports available in: quality_reports\"
            Write-Success "🌐 Advanced report: $reportPath"
            
            # Ask to open report
            $openReport = Read-Host "Open advanced report in browser? (y/n)"
            if ($openReport -eq "y" -or $openReport -eq "Y") {
                Start-Process $reportPath
            }
        } else {
            Write-Error "Automation failed. Check logs for details."
            exit 1
        }
    }
    
} catch {
    Write-Error "Script execution failed: $($_.Exception.Message)"
    Write-Error $_.ScriptStackTrace
    exit 1
}

Write-Success ""
Write-Success "🎉 Advanced Code Quality Automation Complete!"