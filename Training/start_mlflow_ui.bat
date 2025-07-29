@echo off
echo Demarrage de l'interface web MLflow...
echo L'interface sera accessible a l'adresse: http://localhost:5000
echo Appuyez sur Ctrl+C pour arreter le serveur
echo.

cd /d "%~dp0"
python -m mlflow ui --host 0.0.0.0 --port 5000

pause
