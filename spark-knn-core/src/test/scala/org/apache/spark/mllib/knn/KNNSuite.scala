package org.apache.spark.mllib.knn

import org.apache.spark.SharedSparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.{FunSpec, Matchers, FunSuite}

import scala.util.Random

class KNNSuite extends FunSuite with SharedSparkContext with Matchers {

  private[this] val data = (-10 to 10).flatMap(i => (-10 to 10).map(j => Vectors.dense(i, j)))
  private[this] val leafSize = 5

  test("KNNRDD can be constructed") {
    val knn = new KNN(data.size / 10, leafSize, leafSize)
    val knnRDD = knn.run(sc.parallelize(data)).cache()
    //it("should contain all input data") {
    knnRDD.collect().map(_.vector) should contain theSameElementsAs data
    //it("should return an answer that is not too far apart when queried with nearest neighbor")
    val results = knnRDD.query(knnRDD).collect()
    results.size shouldBe data.size

    results.foreach {
      case (p, itr) =>
        itr.size shouldBe 1
        p.vectorWithNorm.fastSquaredDistance(itr.head.vectorWithNorm) should be <= 2.0
    }
  }
}
