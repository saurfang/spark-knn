package org.apache.spark.mllib.knn

import org.apache.spark.SharedSparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.{Matchers, FunSuite}

class KNNSuite extends FunSuite with SharedSparkContext with Matchers {

  test("KNNRDD can be constructed and contain all input data") {
    val data = (-5 to 5).flatMap(i => (-5 to 5).map(j => Vectors.dense(i, j))).map(x => (x, null))
    val knn = new KNN(data.size, 5)
    val knnRDD = knn.run(sc.parallelize(data))
    knnRDD.collect() should contain theSameElementsAs data
  }
}
