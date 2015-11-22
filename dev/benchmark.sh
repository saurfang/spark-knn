#!/usr/bin/env bash

# install sbt and git
curl https://bintray.com/sbt/rpm/rpm | sudo tee /etc/yum.repos.d/bintray-sbt-rpm.repo
sudo yum install sbt git

# download mnsit data (8MM observations)
#wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist8m.bz2
curl -vs https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist8m.bz2 | hadoop fs -put - mnist8m.bz2 &
curl -vs https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 | hadoop fs -put - mnist.bz2

# clone spark-knn
git clone -b benchmark https://github.com/saurfang/spark-knn.git
cd spark-knn
sbt examples/assembly

# test all models for small n
spark-submit --master yarn --num-executors 80 --executor-cores 2 --executor-memory 2500m \
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    2500,5000,7500,10000,20000,40000 mnist.bz2 100

# test tree only for large n
spark-submit --master yarn --num-executors 80 --executor-cores 2 --executor-memory 2500m \
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    80000,160000,320000,640000 mnist8m.bz2 mnist8m.bz2 100 tree

spark-submit --master yarn --num-executors 80 --executor-cores 2 --executor-memory 2500m \
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    1280000,2560000,5120000 mnist8m.bz2 mnist8m.bz2 100 tree

# benchmark horizontal scalability
spark-submit --master yarn --num-executors 5 --executor-cores 2 --executor-memory 2500m \
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    100000 mnist.bz2 5

spark-submit --master yarn --num-executors 10 --executor-cores 2 --executor-memory 2500m \
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    100000 mnist.bz2 10

spark-submit --master yarn --num-executors 20 --executor-cores 2 --executor-memory 2500m \
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    100000 mnist.bz2 25

spark-submit --master yarn --num-executors 40 --executor-cores 2 --executor-memory 2500m \
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    100000 mnist.bz2 50

spark-submit --master yarn --num-executors 40 --executor-cores 4 --executor-memory 5000m \
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    100000 mnist.bz2 100
