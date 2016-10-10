REM to use this, copy this file somewhere and change commands in the "Require Change" section. Define a "yo.cmd" file in your home directory to simply run this file. Thus, to get the aliases, you can get the aliases by typing "~/yo". You can instead automate this by modifying the cmd shortcut to run a command on startup.

REM Note that lines with "REM" in front are comments


REM **********************************
REM General
REM **********************************

doskey np=notepad $*
doskey s.=start .
doskey gr=wgrep -r $1 *
doskey dc=echo Oops. & cd $*
doskey c..=cd ../..

REM **********************************
REM Require Change
REM **********************************

doskey doc=cd C:\Documents

doskey out=$* $g C:\Users\i854584\Documents\out.txt $t notepad C:\Users\i854584\Documents\out.txt
    REM for out, you type "out ./program arg1" and it takes the output of "./program arg1", writes it to an out file, and loads the output in the output file.

doskey al=notepad c:\Users\i854584\Documents\Commands\aliases.cmd

doskey cmds=cd c:\Users\i854584\Documents\Commands\

REM **********************************
REM Other Help
REM **********************************

REM doskey src=cd C:\src
REM set PATH=C:\src\programs\python27;%PATH%
REM doskey sprod=c:\Users\i854584\Documents\Commands\sendProdBuild $1
REM (create own .cmd files)

