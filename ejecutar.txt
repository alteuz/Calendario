@echo off
echo Iniciando la aplicacion CALENDARIO...
set PORTABLE_PYTHON=%~dp0pythonport\WPy64-312100\python\python.exe
if exist "%PORTABLE_PYTHON%" (
    set PYTHON="%PORTABLE_PYTHON%"
) else (
    echo No se encontro Python portable en la ruta esperada. Intentando usar python global...
    set PYTHON=python
)
%PYTHON% -m streamlit run calendario.py
pause
