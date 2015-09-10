package org.apache.spark.mllib.knn

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.scalatest.{Matchers, FunSpec}

class MetricTreeSpec extends FunSpec with Matchers {

  describe("MetricTree") {
    val origin = Vectors.dense(0, 0)
    describe("can be constructed with empty data") {
      val tree = MetricTree.create(IndexedSeq.empty[Vector])
      it("iterator should be empty") {
        tree.iterator shouldBe empty
      }
      it("should return empty when queried") {
        tree.query(origin) shouldBe empty
      }
    }

    describe("without duplicates") {
      val data = (-5 to 5).flatMap(i => (-5 to 5).map(j => Vectors.dense(i, j)))
      List(1, data.size / 2, data.size, data.size * 2).foreach {
        leafSize =>
          describe(s"with leafSize of $leafSize") {
            val tree = MetricTree.create(data)
            import tree._
            it("should have correct size") {
              tree.size shouldBe data.size
              (leftChild.size + rightChild.size) shouldBe tree.size
            }
            it("should return an iterator that goes through all data points") {
              tree.iterator.toIterable should contain theSameElementsAs data
            }
            it("should return vector itself for those in input set") {
              data.foreach(v => tree.query(v).head shouldBe v)
            }
            it("should return nearest neighbors correctly") {
              tree.query(origin, 5) should contain theSameElementsAs Set(
                Vectors.dense(-1, 0),
                Vectors.dense(1, 0),
                Vectors.dense(0, -1),
                Vectors.dense(0, 1),
                Vectors.dense(0, 0)
              )
              tree.query(origin, 9).toSet should contain theSameElementsAs
                (-1 to 1).flatMap(i => (-1 to 1).map(j => Vectors.dense(i, j)))
            }
          }
      }
    }

    describe("for other corner cases") {
      it("queryCost should work on Empty") {
        Empty.queryCost(new KNNCandidates(new VectorWithNorm(origin, 0), 1)) shouldBe 0
      }
    }
  }
}
