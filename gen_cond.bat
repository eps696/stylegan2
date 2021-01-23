@echo off
if "%1"=="" goto help

python src/_genSGAN2_cond.py --model models/%1 --out_dir _out/%~n1 --size %2 --frames %3 ^
%4 %5 %6 %7 %8 %9

ffmpeg -y -v warning -i _out\%~n1\%%06d.jpg -c:v mjpeg -q:v 2 _out/%~n1-%2.avi
rmdir /s /q _out\%~n1

goto end


:help
echo Usage: gen_cond model x-y framecount-transit [--labels z]
echo e.g.:  gen_cond ffhq-1024 1280-720 100-25

:end
