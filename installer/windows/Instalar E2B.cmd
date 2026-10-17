@echo off
setlocal DisableDelayedExpansion
title Instalar E2B QAT - Jev QAT Local
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Instalar.ps1" %*
if errorlevel 1 (
  echo.
  echo No se completo la instalacion. Revisa el mensaje anterior y vuelve a ejecutar este archivo.
  pause
  exit /b 1
)
echo.
echo Listo. Puedes cerrar esta ventana.
