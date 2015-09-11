package org.apache.spark.mllib.knn

import org.apache.spark.SharedSparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.{FunSpec, Matchers, FunSuite}

class KNNSpec extends FunSuite with SharedSparkContext with Matchers {

  test("KNNRDD can be constructed") {
    val data = (-5 to 5).flatMap(i => (-5 to 5).map(j => Vectors.dense(i, j))).map(x => (x, null))
    val leafSize = 5
    val knn = new KNN(data.size, leafSize)
    val knnRDD = knn.run(sc.parallelize(data))
    //it("should only have one element in each partition") {
    knnRDD.mapPartitions(itr => Iterator(itr.size)).collect().foreach(_ should be <= leafSize)
    //it("should contain all input data") {
    knnRDD.collect() should contain theSameElementsAs data
  }
}
