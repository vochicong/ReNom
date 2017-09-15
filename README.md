# ReNom

Documents are available on the ReNom.jp web site.

- http://renom.jp/index.html


## ReNom version 2.1

1. Modified GPU memory allocation
* Modified GPU code of power operation
* Modified GPU code of peephole lstm
* Added soft max activation function
* Added cross entropy loss function
* Fixed lstm temporal backward 
* Fixed momentum sgd


## Requirements

- python2.7, 3.4
- numpy 1.13.0 1.12.1
- pytest 3.0.7
- cython 0.24
- cuda-toolkit 8.0
- cudnn 5.1


## Installation

First clone the ReNom repository.

	git clone https://github.com/ReNom-dev-team/ReNom.git

Then move to the ReNomAd folder, install the module using pip.

	cd ReNom
	pip install -e .

To activate CUDA, you have to build cuda modules before `pip install -e .` 
using following command.

    python setup.py build_ext -if

Please be sure that the environment variable CUDA_HOME is set correctly.

Example:

	$ echo $CUDA_HOME
	/usr/local/cuda-8.0
	

## Precision

If you set an environment variable RENOM_PRECISION=64, 
calculations are performed with float64.

Default case, the precision is float32.


## License

“ReNom” is provided by GRID inc., as subscribed software.  By downloading ReNom, you are agreeing to be bound by our ReNom Subscription agreement between you and GRID inc.
To use ReNom for commercial purposes, you must first obtain a paid license. Please contact us or one of our resellers.  If you are an individual wishing to use ReNom for academic, educational and/or product evaluation purposes, you may use ReNom royalty-free.
The ReNom Subscription agreements are subject to change without notice. You agree to be bound by any such revisions. You are responsible for visiting www.renom.jp to determine the latest terms to which you are bound.
