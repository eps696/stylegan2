@echo off

python src/models_blend.py --out_dir ./ --pkl1 %1 --pkl2 %2 --res %3 --level %4
