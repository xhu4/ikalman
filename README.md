# iKalman

[![GitHub license](https://img.shields.io/apm/l/vim-mode.svg)](https://github.com/xhu4/ikalman/blob/master/LICENSE)
![matlab](https://img.shields.io/badge/python-3-blue.svg)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/xhu4/ikalman/master)

[An interactive jupyter notebook](https://mybinder.org/v2/gh/xhu4/ikalman/master?filepath=ikalman.ipynb)
to illustrate Kalman-Bucy filter.

This is a course project for the course _Stochastic Processes and Applications_, 
in 2018 Fall, at University of Wyoming. So it is heavily written in Stochastic 
language and notations taught in that course. But the interactive plots and the 
notes directly after the plots could be helpful in understanding what _filtering_
and _data_assimilation_ is about.

Run a discrete square dynamic interactively
![](screenshots/dscrt_HD.gif)

A continuous circular dynamic interactively
![](screenshots/cont_HD.gif)

And plot the Kalman Filtering process
![](screenshots/filter_HD.gif)

## Requirements

Instead of running it locally, I'd suggest open it on Binder, 
by just clicking [this link](https://mybinder.org/v2/gh/xhu4/ikalman/master?filepath=ikalman.ipynb).

Or, if you insist, use the `requirements.txt` to build dependencies:
```
pip install -r requirements.txt
```

But there are no fancy requirements. If all the packages below are installed, you should be good to go:

- jupyter
- bqplot
- scipy
- numpy
