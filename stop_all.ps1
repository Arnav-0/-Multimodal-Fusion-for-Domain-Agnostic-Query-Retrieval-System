# Stop all servers - clean shutdown script
Write-Host "=== Stopping All Servers ===" -ForegroundColor Cyan

function Stop-PortProcesses {
    param([int[]]$Ports)
    
    foreach ($port in $Ports) {
        Write-Host "Checking port $port..." -ForegroundColor Yellow
        $connections = netstat -ano | findstr ":$port"
        if ($connections) {
            $pids = $connections | ForEach-Object {
                if ($_ -match '\s+(\d+)\s*$') {
                    $matches[1]
                }
            } | Select-Object -Unique
            
            foreach ($pid in $pids) {
                if ($pid -and $pid -ne "0") {
                    try {
                        $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
                        if ($process) {
                            Write-Host "  Killing process $pid ($($process.ProcessName)) on port $port" -ForegroundColor Red
                            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
                            Start-Sleep -Milliseconds 500
                        }
                    } catch {
                        Write-Host "  Could not kill PID $pid" -ForegroundColor DarkRed
                    }
                }
            }
        }
    }
}

# Kill processes on our ports
Stop-PortProcesses -Ports @(8000, 8001, 8501)

Write-Host "`nAll servers stopped!" -ForegroundColor Green
Write-Host "Run start_unified.ps1 to restart everything.`n" -ForegroundColor Cyan
