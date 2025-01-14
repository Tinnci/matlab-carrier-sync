# 设置脚本所在目录
$scriptDir = "E:\sync"

# 导航到脚本目录
Set-Location $scriptDir

# 运行 MATLAB 脚本并保存输出
matlab -batch "main" > "$scriptDir\matlab_output.log" 2>&1

# 检查 MATLAB 是否成功运行
if ($LASTEXITCODE -eq 0) {
    Write-Output "MATLAB 脚本已成功运行。"
} else {
    Write-Output "MATLAB 脚本运行失败。请检查日志文件。"
}
