@echo off
if "%1"=="" goto help

python src/training/dataset_tool.py --dataset data/%1 --jpg ^
%2 %3 %4 %5 %6 %7 %8 %9
goto end

:help
echo Usage: prepare_dataset imagedir

:end
