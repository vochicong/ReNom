# ReNom

Documents are available on the ReNom.jp web site.

- http://renom.jp/index.html

## Requirements

- python2.7, 3.4
- numpy 1.13.0 1.12.1
- pytest 3.0.7
- cuda-toolkit 8.0
- cudnn 5.1


## Installation

First clone the ReNom repository.

	git clone https://github.com/ReNom-dev-team/ReNom.git

Then move to the ReNomAd folder, install the module using pip.

	cd ReNomAd
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

本ソフトウェア「ReNom」は、株式会社グリッドが提供する有償のソフトウェアとなり、商用目的でのご利用は有償で提供しています。

お客様が個人目的で利用、学術・教育目的で利用、または製品評価の目的で利用する場合（商用目的でない利用）については無償で提供しています。

本ソフトウェアはサブスクリプションでのご提供となり、お客様がサブスクリプションをご利用、もしくは更新をもって発効され、お客様はサブスクリプション契約の内容・条件に同意したものとみなされます。ご利用に必要な有償ライセンスの購入については、当社または当社が指定する販売会社へ所定の手続きの上、ご購入ください。

※サブスクリプション契約の内容については、お客様への事前の予告なく変更される場合があります。本ソフトウェアをご利用される場合、お客様自身の責任と負担において最新契約内容をwww.renom.jpより必ずご確認の上ご利用ください。
