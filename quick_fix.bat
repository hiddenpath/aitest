@echo off
echo 🔧 快速修复AI聊天系统问题...
echo.

echo 1. 创建必要的目录...
if not exist "D:\ai_data" (
    mkdir "D:\ai_data"
    echo ✅ 已创建数据库目录
)

if not exist "static" (
    mkdir "static"
    echo ✅ 已创建静态文件目录
)

echo.
echo 2. 创建基本的静态文件...

REM 创建基本的 index.html
echo ^<!DOCTYPE html^> > static\index.html
echo ^<html^> >> static\index.html
echo ^<head^> >> static\index.html
echo     ^<title^>AI聊天系统^</title^> >> static\index.html
echo     ^<meta charset="utf-8"^> >> static\index.html
echo ^</head^> >> static\index.html
echo ^<body^> >> static\index.html
echo     ^<h1^>AI聊天系统^</h1^> >> static\index.html
echo     ^<p^>系统正在运行...^</p^> >> static\index.html
echo ^</body^> >> static\index.html
echo ^</html^> >> static\index.html

REM 创建基本的 styles.css
echo body { font-family: Arial, sans-serif; margin: 20px; } > static\styles.css
echo h1 { color: #333; } >> static\styles.css

echo ✅ 已创建基本静态文件

echo.
echo 3. 检查并安装依赖...
cargo check
if %ERRORLEVEL% NEQ 0 (
    echo ❌ 依赖检查失败，尝试更新...
    cargo update
)

echo.
echo 4. 清理并重新编译...
cargo clean
cargo build

echo.
echo ✅ 快速修复完成！
echo 现在可以尝试运行: run.bat

pause