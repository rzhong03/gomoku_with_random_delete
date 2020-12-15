@echo off 
set /a ii=0
for /l %%i in (1,1,100) do (
start python game.py 6 5
)

