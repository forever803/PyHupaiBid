@echo off
:loop
echo git %1
git %1

if %ERRORLEVEL% == 1 (
echo errorlevel: %ERRORLEVEL% retry...
goto loop
)
echo errorlevel: %ERRORLEVEL% 

