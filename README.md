# WAIFu: a Weighted-Aligned Implicit Function

What is WAIFu?

This is waifu:

![820721](https://raw.githubusercontent.com/SankHyan24/image1/master/img/820721.jpg)

Just a practice for implement a 3D reconstruction model like PIFu.

## requirements.txt Backup:
```txt
# python 3.8 is recommended.

matplotlib==3.5.2
PyOpenGL==3.1.6 
# use this if OpenGL.error.NullFunctionError: Attempt to call an undefined function glutInit. 
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl  PyOpenGL‑3.1.6‑cp38‑cp38‑win_amd64.whl
torch==1.12.0 
# use this if cuda is not available. 
# https://download.pytorch.org/whl/cu116/torch-1.12.0%2Bcu116-cp38-cp38-win_amd64.whl
torchvision==0.13.0
opencv-python # 4.6.0.66, for example
```

## How to Update "requirements.txt"?
use this command
```shell
pipreqs ./ --encoding=utf8 --force
```

## PRT Issue:
You should install the conda package "pyembree", which is not support for windows using conda.
For windows, plz reference to [this](https://github.com/scopatz/pyembree/issues/14)