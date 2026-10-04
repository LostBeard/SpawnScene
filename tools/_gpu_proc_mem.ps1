# Samples Chrome's GPU process memory (private bytes, working set, pagefile) and total VRAM every 2 s, to see what a
# "GPU process died due to out of memory" device loss runs out of. Usage: pwsh tools/_gpu_proc_mem.ps1 <out.txt>
param([string]$Out)
while ($true) {
    $gpu = Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | ? { $_.CommandLine -like '*--type=gpu-process*' -and $_.CommandLine -like '*spawnscene-harness*' }
    $vram = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -join ''
    $os = Get-CimInstance Win32_OperatingSystem
    foreach ($g in $gpu) {
        $p = Get-Process -Id $g.ProcessId -ErrorAction SilentlyContinue
        if ($p) {
            "{0} pid {1} private {2:N0} MB ws {3:N0} MB vram {4} MB ramfree {5:N0} MB" -f (Get-Date -Format HH:mm:ss), $p.Id,
                ($p.PrivateMemorySize64 / 1MB), ($p.WorkingSet64 / 1MB), $vram, ($os.FreePhysicalMemory / 1KB) | Add-Content $Out
        }
    }
    Start-Sleep 2
}
