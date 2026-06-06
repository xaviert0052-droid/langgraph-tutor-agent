@echo off
cd /d "%~dp0.."
echo.
echo  === 同步到 GitHub ===
echo.

:: 暂存
git add -A

:: 看改了啥
git status --short

echo.
set /p msg=提交说明（直接回车=自动生成）: 

if "%msg%"=="" (
  git commit -m "chore: update %date% %time%"
) else (
  git commit -m "%msg%"
)

echo.
git push

echo.
echo  === 完成 ===
pause
