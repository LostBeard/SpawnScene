@echo off

set configuration=Release
set outputPath=bin\Publish%configuration%

REM build default
echo "Creating %configuration% publish build"
rmdir /Q /S "%outputPath%"
dotnet publish --nologo --configuration %configuration% --output "%outputPath%"

echo "Build complete."

