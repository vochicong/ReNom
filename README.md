# ReNom

Documents are available on the ReNom.jp web site.

- http://renom.jp/index.html

## ReNom version 2.6.2
- http://renom.jp/rsts/renomdl/main.html

#### Changes from 2.5

Please refer to `changes` at renom.jp.

- Improved performance.
- Add function `curand_set_seed` to determine the curand random number generator's seed.
- Add argument `ignore_bias` to all parametrized class.
- Add argument `reduce_sum` to all loss function class.
- Add `Weight Normalize`.
- Add `Layer Normalize`.
- Add `GPUDistributor`.
- Add `Gru`.
- Add `Convlution 3d, nd`.
- Add `Max, Avg Pooling 3d, nd`.
- Add `Unpooling 2d`.

#### Changes from 2.6.1
- Bug fix of the class ConvNd, PoolNd(To be acceptable 1d data.)
- Bug fix of the class Unpool2d, Unpoolnd2.


## Requirements

- python2.7, 3.4
- numpy 1.13.0, 1.12.1
- pytest 3.0.7
- cython 0.24
- cuda-toolkit 8.0, 9.1
- cudnn 5.1, 6.0, 7.1
- matplotlib 2.0.2
- pandas 0.20.3
- scikit-learn 0.18.2
- scipy 0.19.0
- tqdm 4.19.4

## Installation

First clone the ReNom repository.

	git clone https://github.com/ReNom-dev-team/ReNom.git

Then move to the ReNom folder, install the module using pip.

	cd ReNom
	pip install -e .

To activate CUDA, you have to build cuda modules before `pip install -e .` 
using following command.

    python setup.py build_ext -if

Please be sure that the environment variable CUDA_HOME is set correctly.

Example:

	$ echo $CUDA_HOME
	/usr/local/cuda-9.1
	

## Precision

If you set an environment variable RENOM_PRECISION=64, 
calculations are performed with float64.

Default case, the precision is float32.

## Limit of tensor dimension size.
In ReNom version >= 2.4, only tensors that have less than 6 dimension size can be operated.


## License

“ReNom” is provided by GRID inc., as subscribed software.  By downloading ReNom, you are agreeing to be bound by our ReNom Subscription agreement between you and GRID inc.
To use ReNom for commercial purposes, you must first obtain a paid license. Please contact us or one of our resellers.  If you are an individual wishing to use ReNom for academic, educational and/or product evaluation purposes, you may use ReNom royalty-free.
The ReNom Subscription agreements are subject to change without notice. You agree to be bound by any such revisions. You are responsible for visiting www.renom.jp to determine the latest terms to which you are bound.
