@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] venv not found: .venv
    echo Run: python -m venv .venv
    echo Then: .venv\Scripts\python.exe -m pip install -r src\requirements.txt
    pause
    exit /b 1
)

set PY=.venv\Scripts\python.exe

echo [1/5] Q1 extract frames
%PY% src\q1_extract_frames.py || goto :err

echo [2/5] Q2 orange mask
%PY% src\q2_orange_mask.py || goto :err

echo [3/5] Q3 filter and morphology compare
%PY% src\q3_filter_morph.py || goto :err

echo [4/5] Q4 four corners on sample frames
%PY% src\q4_find_corners.py || goto :err

echo [5/5] Q5 annotate whole video
%PY% src\q5_annotate_video.py || goto :err

echo.
echo [DONE] all five steps finished, results in outputs\
pause
exit /b 0

:err
echo.
echo [FAIL] last step returned an error
pause
exit /b 1
