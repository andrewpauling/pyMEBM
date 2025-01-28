# pyMEBM
Python version of the Moist Energy Balance Model from Roe et al., (2015)

Rewrote from a Matlab version of the model provided for ATM S 591: Climate Dynamics.
Currently two options for the model:
* Climatology EBM which computes the equilibrium temperature profile for a given solar forcing and diffusivity.
* Perturbation EBM which computes the temperature response to a given combination of forcing and feedbacks.

Seasonal EBM is still a work in progress.

To create a python environment with all the necessary packages:

```mamba env create -f environment.yml```

Then to add the code in pyMEBM so that it can be imported:

```
mamba activate pymebm
pip install e .
```


This is still very much a **work in progress**, feel free to contact me or post an issue if you are interested.
