# Graph Wavelets via Sparse Cuts

Implementation of graph wavelets via sparse cuts with some baselines, datasets and evaluation.

Evaluation is performed using python notebooks.

## Scalability and approximation experiments:

<https://nbviewer.jupyter.org/github/arleilps/sparse-wavelets/blob/master/synthetic-data.ipynb>

## Compression experiments:

<https://nbviewer.jupyter.org/github/arleilps/sparse-wavelets/blob/master/compression-experiments.ipynb>

For more details, see the paper:<br>
[Graph Wavelets via Sparse Cuts](http://arxiv.org/abs/1602.03320) <br>
Arlei Silva, Xuan-Hong Dang, Prithwish Basu, Ambuj K Singh, Ananthram Swami<br>
ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016 (to appear).

Arlei Silva (arlei@cs.ucsb.edu)

# Requirements

- python3
    - use of math.log2()

```sh
sudo apt install ipython3
sudo apt install python3-notebook
sudo apt install python3-networkx
sudo apt install python3-scipy
sudo apt install python3-matplotlib
sudo apt install python3-pywt
sudo apt install python3-sklearn
sudo apt install python3-pandas
pip3 install statistics
sudo pip3 install statsmodels
```

# Run

```sh
ipython3 nbconvert -to python *.ipynb
python3 <generated python script>
```

# Known Issues

-  NONE
