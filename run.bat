@echo off
rem Runs the project with the Python from this folder's venv, so it works
rem even in a fresh terminal where the venv was not activated.
rem Usage:  run --output game      (any main.py options work)
"%~dp0venv\Scripts\python.exe" "%~dp0main.py" %*
