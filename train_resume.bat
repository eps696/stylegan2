@echo off

python src/train.py --dataset data/%1 --resume train/%2 ^
--jpg_data ^
%3 %4 %5 %6 %7 %8 %9 

