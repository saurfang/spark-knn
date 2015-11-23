# kNN Benchmark



We use [MNIST data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html) to benchmark our kNN implementation. Recall our kNN implementation is based on hybrid spill tree and more details can be found at the main
README.

For tests with less than 60k observations, we use the regular MNIST. For those with more than 60k
observations, we opted for the *mnist8m* processed dataset.

## kNN on local Spark
We first compare kNN runtime performace on Spark local mode. 
![](benchmark_files/figure-html/pre-run local-1.png) 
While the spill-tree implementation has much larger overhead, the savings on the search efficiency quickly
trumps the naive brute-force approach when n gets larger.

## kNN on local R
For perspective, we also ran the kNN using RANN in R which is based on KD-tree and knn in class package which is brute force based.

Note: all Spark benchmark is average of three runs while all R local benchmark numbers are even less scientific with a single run instead.




![](benchmark_files/figure-html/local-plot-1.png) 

## kNN on Spark Clsuter
Next we test our kNN on AWS 10 c3.4xlarge nodes cluster (160 cores in total).

Note for larger n, we only ran the algorithm using spill tree due to much longer runtime for naive approach.



![](benchmark_files/figure-html/cluster-plot-1.png) 

Notice the y-axis is on log scale.

## Horizontal Scalability

Finally we will examine how the algorithm scales with the number of cores. Again this is using AWS c3.4xlarge nodes.


![](benchmark_files/figure-html/horizontal-plot-1.png) 

Ideally we want the algorithm to scale linearly and we can see our kNN implementation scales quite linearly up to 80 cores The diminishing returns is likely attributed to the low number of observations. For 160 cores, each core is merely responsible for 375 observations on average. In practice, we were able to scale the implementation on hundrends of millions of observations much better with thousands of cores

Note: The naive implementation scales much poorly because some tasks randomly decide to read from network.

