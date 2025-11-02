Write-Host "Configuring environment for GPU and servers..." -ForegroundColor Yellow

# Load .env into a hashtable
$envVars = @{}
if (Test-Path .env) {
	Write-Host "Loading .env" -ForegroundColor DarkGray
	Get-Content .env | ForEach-Object {
		if (-not [string]::IsNullOrWhiteSpace($_) -and -not $_.Trim().StartsWith('#')) {
			$parts = $_.Split('=',2)
			if ($parts.Length -eq 2) {
				$key = $parts[0].Trim(); $val = $parts[1].Trim()
				if ($key) { 
					$envVars[$key] = $val
					Set-Item -Path Env:$key -Value $val -ErrorAction SilentlyContinue
				}
			}
		}
	}
}

# Override critical settings for GPU
$envVars['MODEL_DEVICE'] = 'cuda'
$envVars['MODEL_DEVICE_ID'] = '0'
$envVars['FAISS_USE_GPU'] = 'true'
$envVars['GEMINI_INCLUDE_IMAGES'] = 'true'

# Set in current process too
$env:MODEL_DEVICE = "cuda"
$env:MODEL_DEVICE_ID = "0"
$env:FAISS_USE_GPU = "true"
$env:GEMINI_INCLUDE_IMAGES = "true"

function Kill-Port($port) {
	$lines = netstat -ano | Select-String ":$port\s+" | ForEach-Object { $_.ToString() }
	$pids = @()
	foreach ($ln in $lines) {
		$parts = $ln -split "\s+"
		if ($parts.Length -ge 5) {
			$pidVal = $parts[$parts.Length - 1]
			if ($pidVal -match '^[0-9]+$') { $pids += [int]$pidVal }
		}
	}
	$pids = $pids | Sort-Object -Unique
	foreach ($procId in $pids) {
		try { Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue } catch {}
	}
}

# Prefer venv python if exists
$pythonExe = "python"
if (Test-Path ".\.venv\Scripts\python.exe") { $pythonExe = ".\.venv\Scripts\python.exe" }

Write-Host "Freeing ports 8000/8001/8501..." -ForegroundColor Yellow
Kill-Port 8000
Kill-Port 8001
Kill-Port 8501
Start-Sleep -Seconds 2

# Build environment string for child processes
$envString = ""
foreach ($key in $envVars.Keys) {
	$val = $envVars[$key]
	$envString += "`$env:$key='$val'; "
}

Write-Host "Starting Model Server on port 8001..." -ForegroundColor Green
$modelCmd = "cd 'd:\Final project'; $envString `"$pythonExe`" -m uvicorn model_server:app --host 127.0.0.1 --port 8001"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $modelCmd

Start-Sleep -Seconds 8

Write-Host "Starting Unified Fusion API on port 8000..." -ForegroundColor Green
$apiCmd = "cd 'd:\Final project'; $envString `"$pythonExe`" -m uvicorn main_api:app --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $apiCmd

Start-Sleep -Seconds 3

Write-Host "Starting Streamlit dashboard on port 8501..." -ForegroundColor Green
$streamlitCmd = "cd 'd:\Final project'; $envString `"$pythonExe`" -m streamlit run app.py --server.headless true --server.port 8501"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $streamlitCmd

Write-Host ""
Write-Host "Unified server running:" -ForegroundColor Yellow
Write-Host "  Model Server:    http://localhost:8001/health" -ForegroundColor Cyan
Write-Host "  Unified API:     http://localhost:8000/hackrx/run" -ForegroundColor Cyan
Write-Host "  Streamlit UI:    http://localhost:8501" -ForegroundColor Cyan