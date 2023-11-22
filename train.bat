set root=C:\ProgramData\miniconda3

call %root%\Scripts\activate.bat %root%

call activate gpu-rl

call pip list

call python train.py

pause
