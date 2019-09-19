# :high_brightness: SPNGP

This a Python implementation of the SPNGP model proposed in
[Learning Deep Mixtures of Gaussian Process Experts Using Sum-Product Networks](https://arxiv.org/abs/1809.04400) (arXiv:1809.04400) with some tweaks and improvements.

## :wrench: Tweaks and improvements 

After the initial "vanilla" implementation it was quickly discovered that the network performed best when splits resulted in balanced regions. If some of the regions only had only a handful of instances and others had a lot, performance decayed substantially. This aligned with our intuition - if a GP was trained only on a handful of instances, it was naturally less precise at inference. To solve this problem, we developed a number of improvements to the algorithm. Although simple, those are effective contributions and to the best of our knowledge, they have not been published before.

* Instead of splitting on equidistant locations in the input space, we split on quantiles.
* Instead of committing to splits in same dimension for all children of a Sum node, we allow splits to cut through different dimensions at the same level of recursion.
* Instead of randomly choosing dimensions when splitting, we prioritize dimensions where data is more uniformly distributed. To quantify uniformity we use entropy (after binning the series).

## :running: Running
Apart from the usual `numpy` and `pandas`, this code also depends on the excellent [GPyTorch library](https://gpytorch.ai) to handle the actual GP's. To evaluate the model on the datasets from the SPNGP paper, run:
```python
python cccp-spngp.py
python energy-spngp.py
python concrete-spngp.py
```

## :bar_chart: Results
Following is a summary of the results:

| Dataset   | Nvars | N    |  RMSE (Ours)  | RMSE (Trapp) |
| --------- | ----- | ---- | ------------- | ------------ |
| energy    | 8     |  768 | **1.25**          | 2.07         |
| concrete  | 8     | 1030 | **4.84**          | 6.25         |
| ccpp      | 4     | 9568 | **3.68**          | 4.11         |

## :mortar_board: Credits
This code is a part of a MSc thesis written by Yevgen "Eugene" Zainchkovskyy at DTU Compute, department of Applied Mathematics and Computer Science at the Technical University of Denmark with an industrial partner Alipes Capital ApS. The work was carried out under supervision of Ole Winther, Professor at Section for Cognitive Systems, DTU Compute and Carsten Stahlhut, PhD, Principal Data Scientist, Novo Nordisk A/S (former Head of Quants at Alipes Capital). 

A very special gratitude goes to Martin Trapp ([@trappmartin](https://github.com/trappmartin)) for the SPNGP model and countless emails in which he helped the author of this code with explanations and understanding of the underlying details. 
