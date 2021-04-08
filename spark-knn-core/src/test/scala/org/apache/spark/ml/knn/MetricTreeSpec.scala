package org.apache.spark.ml.knn

import org.apache.spark.ml.knn.KNN.{EuclideanDistanceMetric, RowWithVector, VectorWithNorm}
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

class MetricTreeSpec extends AnyFunSpec with Matchers {

  describe("MetricTree") {
    val distanceMetric = EuclideanDistanceMetric
    val origin = Vectors.dense(0, 0)
    describe("can be constructed with empty data") {
      val tree = MetricTree.build(IndexedSeq.empty[RowWithVector], distanceMetric=distanceMetric)
      it("iterator should be empty") {
        tree.iterator shouldBe empty
      }
      it("should return empty when queried") {
        tree.query(origin).isEmpty shouldBe true
      }
      it("should have zero leaf") {
        tree.leafCount shouldBe 0
      }
    }

    describe("without duplicates") {
      val data = (-5 to 5).flatMap(i => (-5 to 5).map(j => new RowWithVector(Vectors.dense(i, j), null)))
      List(1, data.size / 2, data.size, data.size * 2).foreach {
        leafSize =>
          describe(s"with leafSize of $leafSize") {
            val tree = MetricTree.build(data, leafSize, distanceMetric=distanceMetric)
            it("should have correct size") {
              tree.size shouldBe data.size
            }
            it("should return an iterator that goes through all data points") {
              tree.iterator.toIterable should contain theSameElementsAs data
            }
            it("should return vector itself for those in input set") {
              data.foreach(v => tree.query(v.vector, 1).head._1 shouldBe v)
            }
            it("should return nearest neighbors correctly") {
              tree.query(origin, 5).map(_._1.vector.vector) should contain theSameElementsAs Set(
                Vectors.dense(-1, 0),
                Vectors.dense(1, 0),
                Vectors.dense(0, -1),
                Vectors.dense(0, 1),
                Vectors.dense(0, 0)
              )
              tree.query(origin, 9).map(_._1.vector.vector) should contain theSameElementsAs
                (-1 to 1).flatMap(i => (-1 to 1).map(j => Vectors.dense(i, j)))
            }
            it("should have correct number of leaves") {
              tree.leafCount shouldBe (tree.size / leafSize.toDouble).ceil
            }
            it("all points should fall with radius of pivot") {
              def check(tree: Tree): Unit = {
                tree.iterator.foreach(node=> distanceMetric.fastDistance(node.vector, tree.pivot) <= tree.radius)
                tree match {
                  case t: MetricTree =>
                    check(t.leftChild)
                    check(t.rightChild)
                  case _ =>
                }
              }
              check(tree)
            }
          }
      }
    }

    describe("with duplicates") {
      val data = (Vectors.dense(2.0, 0.0) +: Array.fill(5)(Vectors.dense(0.0, 1.0))).map(new RowWithVector(_, null))
      val tree = MetricTree.build(data, distanceMetric=distanceMetric)
      it("should have 2 leaves") {
        tree.leafCount shouldBe 2
      }
      it("should return all available duplicated candidates") {
        val res = tree.query(origin, 5).map(_._1.vector.vector)
        res.size shouldBe 5
        res.toSet should contain theSameElementsAs Array(Vectors.dense(0.0, 1.0))
      }
    }

    describe("for other corner cases") {
      it("queryCost should work on Empty") {
        Empty(distanceMetric).distance(new KNNCandidates(new VectorWithNorm(origin), 1, distanceMetric)) shouldBe 0
        Empty(distanceMetric).distance(new VectorWithNorm(origin)) shouldBe 0
      }
    }
  }
}
