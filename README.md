# iKalman

[![GitHub license](https://img.shields.io/apm/l/vim-mode.svg)](https://github.com/xhu4/ikalman/blob/master/LICENSE)
![matlab](https://img.shields.io/badge/python-3-blue.svg)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/xhu4/ikalman/master)

An [interactive](https://mybinder.org/v2/gh/xhu4/ikalman/master?filepath=ikalman.ipynb)
jupyter notebook to illustrate Kalman-Bucy filter.

Run a discrete square dynamic interactively
![](screenshots/dscrt_HD.gif)

A continuous circular dynamic interactively
![](screenshots/cont_HD.gif)

And plot the Kalman Filtering process
![](screenshots/filter_HD.gif)

## Requirements

Instead of running it locally, I'd suggest open it on Binder, 
by just clicking [this link](https://mybinder.org/v2/gh/xhu4/ikalman/master?filepath=ikalman.ipynb).

Or, if you insist, use the `environment.yml` to create a conda environment.

But there are no fancy requirements. If all the packages below are installed, you should be good to go:

- jupyter
- bqplot
- scipy
- numpy
