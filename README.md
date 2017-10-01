# Graph Wavelets via Sparse Cuts

Implementation of graph wavelets via sparse cuts with some baselines, datasets and evaluation.

Evaluation is performed using IPython Notebook.

After code review some results may differ from those presented in the paper.

Scalability and approximation experiments:
-----------------------
https://nbviewer.jupyter.org/github/arleilps/sparse-wavelets/blob/master/synthetic-data.ipynb

Compression experiments:
-----------------------
https://nbviewer.jupyter.org/github/arleilps/sparse-wavelets/blob/master/compression-experiments.ipynb

Testing:
------
At the moment there is only one doctest in lib/experiments.py. To run the test
you should use python version 3.4 or 3.5, NetworkX 1.11 and set PYTHONHASHSEED=0.
This conditions are used to constrain the behaviour of the NetworkX function
fiedler_vector(). Once ready, enter the following command:
```
python -m doctest lib/experiments.py -v
```

List of supported Python versions:
------------------
<ul>
<li>2.7</li>
<li>3.4</li>
<li>3.5</li>
<li>3.6</li>
</ul>

<br />

For more details, see the paper:  
[Graph Wavelets via Sparse Cuts ](http://arxiv.org/abs/1602.03320 "")  
Arlei Silva, Xuan-Hong Dang, Prithwish Basu, Ambuj K Singh, Ananthram Swami  
ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016.

Arlei Silva (arlei@cs.ucsb.edu)
