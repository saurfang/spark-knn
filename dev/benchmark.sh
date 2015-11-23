#!/usr/bin/env bash

# install sbt and git
curl https://bintray.com/sbt/rpm/rpm | sudo tee /etc/yum.repos.d/bintray-sbt-rpm.repo
sudo yum install sbt git

# download mnsit data (8MM observations)
#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist8m.bz2
curl -vs https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 | hadoop fs -put - mnist.bz2
curl -vs https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist8m.bz2 | bzcat | hadoop fs -put - mnist8m

# clone spark-knn
git clone -b benchmark https://github.com/saurfang/spark-knn.git
cd spark-knn
sbt examples/assembly

# test all models for small n
#knn: WrappedArray(7342.333333333333, 4962.666666666666, 5370.0, 5151.333333333333, 6091.333333333333, 8506.666666666666) / WrappedArray(6017.333333333333, 7072.0, 8856.0, 9742.666666666666, 15817.0, 32105.0)
#naive: WrappedArray(1023.6666666666666, 627.6666666666666, 786.3333333333333, 797.3333333333333, 1201.0, 1873.3333333333333) / WrappedArray(4116.333333333333, 4601.666666666666, 5782.0, 7866.666666666666, 19555.666666666664, 70148.33333333333)
spark-submit --master yarn --num-executors 20 --executor-cores 4 --executor-memory 5000m \
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    2500,5000,7500,10000,20000,40000 mnist.bz2 20

# test tree only for large n

#knn: WrappedArray(19883.333333333332, 28345.0, 50083.0) / WrappedArray(63017.33333333333, 189466.66666666666, 641681.3333333333)
spark-submit --master yarn --num-executors 20 --executor-cores 8 --executor-memory 10000m \
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    80000,160000,320000 mnist8m 150 tree

spark-submit --master yarn --num-executors 20 --executor-cores 8 --executor-memory 10000m \
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    640000,1280000 mnist8m 200 tree

spark-submit --master yarn --num-executors 20 --executor-cores 8 --executor-memory 10000m \
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    2560000,5120000 mnist8m 200 tree

# benchmark horizontal scalability

#knn: WrappedArray(15343.666666666666) / WrappedArray(277521.6666666666)
#naive: WrappedArray(3510.333333333333) / WrappedArray(777180.0)
spark-submit --master yarn --num-executors 10 --executor-cores 2 --executor-memory 2500m \
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    100000 mnist.bz2 10

#knn: WrappedArray(15357.0) / WrappedArray(127080.33333333333)
#naive: WrappedArray(3717.0) / WrappedArray(308656.66666666666)
spark-submit --master yarn --num-executors 20 --executor-cores 2 --executor-memory 2500m \
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    100000 mnist.bz2 25

#knn: WrappedArray(13953.0) / WrappedArray(61519.0)
#naive: WrappedArray(2677.333333333333) / WrappedArray(201512.3333333333)
spark-submit --master yarn --num-executors 40 --executor-cores 2 --executor-memory 2500m \
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    100000 mnist.bz2 50

#knn: WrappedArray(14890.666666666666) / WrappedArray(40220.33333333333)
#naive: WrappedArray(2776.0) / WrappedArray(175310.0)
spark-submit --master yarn --num-executors 20 --executor-cores 8 --executor-memory 10000m \
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    100000 mnist.bz2 100
