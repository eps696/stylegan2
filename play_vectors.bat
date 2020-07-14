@echo off
if "%1"=="" goto help

python src/_play_vector.py --model models/%1 --npy_file _in/%2 --vector_dir _in/%3 ^
%4 %5 %6 %7 %8 %9

ffmpeg -y -v warning -r 25 -i _out\ttt\%%05d.jpg -c:v mjpeg -q:v 2 %~n1-%~n3.avi
rmdir /s /q _out\ttt

goto end 

:help
echo Usage: play_vector model latentsdir vector
echo  e.g.: play_vector ffhq-1024-f npy age.npy

:end