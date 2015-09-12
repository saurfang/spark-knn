package org.apache.spark.mllib.knn

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.scalatest.{Matchers, FunSpec}

class MetricTreeSpec extends FunSpec with Matchers {

  describe("MetricTree") {
    val origin = Vectors.dense(0, 0)
    describe("can be constructed with empty data") {
      val tree = MetricTree(IndexedSeq.empty[hasVector])
      it("iterator should be empty") {
        tree.iterator shouldBe empty
      }
      it("should return empty when queried") {
        tree.query(origin) shouldBe empty
      }
      it("should have zero leaf") {
        tree.leafCount shouldBe 0
      }
    }

    describe("without duplicates") {
      val data = (-5 to 5).flatMap(i => (-5 to 5).map(j => hasVector(Vectors.dense(i, j))))
      List(1, data.size / 2, data.size, data.size * 2).foreach {
        leafSize =>
          describe(s"with leafSize of $leafSize") {
            val tree = MetricTree(data, leafSize)
            it("should have correct size") {
              tree.size shouldBe data.size
            }
            it("should return an iterator that goes through all data points") {
              tree.iterator.toIterable should contain theSameElementsAs data
            }
            it("should return vector itself for those in input set") {
              data.foreach(v => tree.query(v.vector).head shouldBe v)
            }
            it("should return nearest neighbors correctly") {
              tree.query(origin, 5).map(_.vector) should contain theSameElementsAs Set(
                Vectors.dense(-1, 0),
                Vectors.dense(1, 0),
                Vectors.dense(0, -1),
                Vectors.dense(0, 1),
                Vectors.dense(0, 0)
              )
              tree.query(origin, 9).map(_.vector) should contain theSameElementsAs
                (-1 to 1).flatMap(i => (-1 to 1).map(j => Vectors.dense(i, j)))
            }
            it("should have correct number of leaves") {
              tree.leafCount shouldBe (tree.size / leafSize.toDouble).ceil
            }
          }
      }
    }

    describe("with duplicates") {
      val data = (Vectors.dense(2.0, 0.0) +: Array.fill(5)(Vectors.dense(0.0, 1.0))).map(hasVector.apply)
      val tree = MetricTree(data)
      it("should have 2 leaves") {
        tree.leafCount shouldBe 2
      }
      it("should return all available duplicated candidates") {
        val res = tree.query(origin, 5).map(_.vector)
        res.size shouldBe 5
        res.toSet should contain theSameElementsAs Array(Vectors.dense(0.0, 1.0))
      }
    }

    describe("for other corner cases") {
      it("queryCost should work on Empty") {
        Empty.queryCost(new KNNCandidates(new VectorWithNorm(origin, 0), 1)) shouldBe 0
        Empty.queryCost(new VectorWithNorm(origin, 0)) shouldBe 0
      }
    }
  }
}
