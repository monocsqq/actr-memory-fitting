@echo off

rem initialize counter
set counter=0

rem execute command for 100 times
:loop
if %counter%==100 goto end
    rem port num from 3101 to 3200
    set /a num=3101+%counter%
    set command=sbcl --eval "(defparameter *given-port* %num%)" --load "./actr7.x/load-act-r.lisp" --load "./actr7.x/tutorial/unit5/grouped-model-thousand.lisp"
    
    rem execute command
    start "" cmd /c %command%
    
	rem set timeout
	timeout /t 1 /nobreak >nul
	
    rem increment counter
    set /a counter+=1
goto loop

:end
