package org.apache.spark.mllib.knn

import org.apache.spark.SharedSparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.{FunSpec, Matchers, FunSuite}

class KNNSuite extends FunSuite with SharedSparkContext with Matchers {

  test("KNNRDD can be constructed") {
    val data = (-10 to 10).flatMap(i => (-10 to 10).map(j => Vectors.dense(i, j)))
    val leafSize = 5
    val knn = new KNN(data.size / 10, leafSize, leafSize)
    val knnRDD = knn.run(sc.parallelize(data)).cache()
    //it("should contain all input data") {
    knnRDD.collect().map(_.vector) should contain theSameElementsAs data
    //it("should return itself when queried with nearest neighbor")
    val results = knnRDD.query(knnRDD).collect()
    results.size shouldBe data.size

    results.foreach {
      case (p, itr) =>
        itr.size shouldBe 1
        p shouldBe itr.head
    }
  }
}
