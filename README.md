# NTUT-tutorial-1
A tutorial that shows you how to control different length when we batch multiple datasets, and how to put into your RNN  model.
## Usage
```
python3 run.py
```
### model.py
> RNN structure, padding function, and train/test operate in here.
### runy.py
> Before training(e.g. infer vectors), using train/test in here.
### Pad.py
> Easy sample for present how to padding.
## Data description

[[[100, 100...], y0],  
 [[100, 100...], y1],  
 [[100, 100...], y2],  
 [[100, 100...], yn]]  
 
 
