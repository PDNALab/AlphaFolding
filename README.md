# AlphaFolding


To run locally, simply download [ColabDesign](https://github.com/sokrypton/ColabDesign) and execute the `alphafolding.py` script with the following:

``python alphafolding.py -seq $(tail -n 1 seq.fasta) -recycle 1 -iter 50 -model_name model_1_ptm``

A comprehensive guide about reproduction instructions and choice of parameters can be found in [our colab notebook](https://github.com/PDNALab/AlphaFolding/blob/main/alphafolding.ipynb).
