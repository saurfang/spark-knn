#!/usr/bin/env bash

# install sbt
echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 642AC823
sudo apt-get update
sudo apt-get install sbt

# download mnsit data (8MM observations)
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist8m.bz2

# clone spark-knn
git clone -b benchmark https://github.com/saurfang/spark-knn.git
cd spark-knn
sbt examples/assembly
cd ..

# test all models for small n
spark-submit --master yarn --num-executors 80 --executor-cores 2 --executor-memory 3G \
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn/spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    mnist8m.bz2 2500,5000,7500,10000,20000,40000 100

# test tree only for large n
spark-submit --master yarn --num-executors 80 --executor-cores 2 --executor-memory 3G \
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn/spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    mnist8m.bz2 80000,160000,320000,640000 100 tree

spark-submit --master yarn --num-executors 80 --executor-cores 2 --executor-memory 3G \ 
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn/spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    mnist8m.bz2 1280000,2560000,5120000 100 tree

# benchmark horizontal scalability
spark-submit --master yarn --num-executors 5 --executor-cores 2 --executor-memory 3G \ 
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn/spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    mnist8m.bz2 100000 10

spark-submit --master yarn --num-executors 10 --executor-cores 2 --executor-memory 3G \ 
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn/spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    mnist8m.bz2 100000 20

spark-submit --master yarn --num-executors 20 --executor-cores 2 --executor-memory 3G \ 
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn/spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    mnist8m.bz2 100000 40

spark-submit --master yarn --num-executors 40 --executor-cores 2 --executor-memory 3G \ 
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn/spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    mnist8m.bz2 100000 80

spark-submit --master yarn --num-executors 80 --executor-cores 2 --executor-memory 3G \ 
    --class com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark \
    spark-knn/spark-knn-examples/target/scala-2.10/spark-knn-examples-assembly-0.1-SNAPSHOT.jar \
    mnist8m.bz2 100000 160
