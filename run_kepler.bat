@echo off 
echo Activating the environment, and launching it on the Kepler Orbit Viewer,... 
call conda activate astrodata 
cd /d "%~dp0" 
python exoplanet_analyzer.py 
pause
