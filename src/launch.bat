@echo off

echo PREPARING MODULE
python prepare.py

echo TRAINING MODULE
python train.py 

echo EVALUATING MODULE 
python evaluate.py 

echo DONE