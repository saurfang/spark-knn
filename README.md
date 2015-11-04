#spark-knn

[![Build Status](https://travis-ci.org/saurfang/spark-knn.svg)](https://travis-ci.org/saurfang/spark-knn)
[![codecov.io](http://codecov.io/github/saurfang/spark-knn/coverage.svg?branch=master)](http://codecov.io/github/saurfang/spark-knn?branch=master)

WIP...

k-Nearest Neighbors algorithm (k-NN) implemented on Apache Spark. This uses a hybrid spill tree approach to
achieve high accuracy and search efficiency. 

## KNNClassifier

```scala
//read in raw label and features
val training = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt").toDF()

val knn = new KNNClassifier()
  .setTopTreeSize(training.count().toInt / 500)
  .setK(10)

val knnModel = knn.fit(training)

val predicted = knn.transform(training)
```

## KNNRegression

```scala
//read in raw label and features
val training = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt").toDF()

val knn = new KNNRegression()
  .setTopTreeSize(training.count().toInt / 500)
  .setK(10)

val knnModel = knn.fit(training)

val predicted = knn.transform(training)
```


## Credits

- Liu, Ting, et al. 
"An investigation of practical approximate nearest neighbor algorithms." 
Advances in neural information processing systems. 2004.

- Liu, Ting, Charles Rosenberg, and Henry Rowley. 
"Clustering billions of images with large scale nearest neighbor search." 
Applications of Computer Vision, 2007. WACV'07. IEEE Workshop on. IEEE, 2007.
