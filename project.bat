@echo off

python src/project_latent.py --model=models/%1 --in_dir=_in/%2 --out_dir=_out/proj/%2 ^
%3 %4 %5 %6 %7 %8 %9

