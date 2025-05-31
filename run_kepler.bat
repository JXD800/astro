@echo off
echo Activating environment and launching Kepler Orbit Viewer...
call conda activate astrodata
cd /d "%~dp0"
python kepler_orbit_final_with_animation.py
pause
