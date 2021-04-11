package org.apache.spark.ml.knn

import org.apache.spark.ml.knn.KNN.RowWithVector
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

class SpillTreeSpec extends AnyFunSpec with Matchers {
  describe("SpillTree") {
    val distanceMetric = EuclideanDistanceMetric
    val origin = Vectors.dense(0, 0)
    describe("can be constructed with empty data") {
      val tree = SpillTree.build(IndexedSeq.empty[RowWithVector], tau = 0.0, distanceMetric=distanceMetric)
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

    describe("with equidistant points on a circle") {
      val n = 12
      val points = (1 to n).map {
        i => new RowWithVector(Vectors.dense(math.sin(2 * math.Pi * i / n), math.cos(2 * math.Pi * i / n)), null)
      }
      val leafSize = n / 4
      describe("built with tau = 0.0") {
        val tree = SpillTree.build(points, leafSize = leafSize, tau = 0.0, distanceMetric=distanceMetric)
        it("should have correct size") {
          tree.size shouldBe points.size
        }
        it("should return an iterator that goes through all data points") {
          tree.iterator.toIterable should contain theSameElementsAs points
        }
        it("can return more than min leaf size") {
          val k = leafSize + 5
          points.foreach(v => tree.query(v.vector, k).size shouldBe k)
        }
      }
      describe("built with tau = 0.5") {
        val tree = SpillTree.build(points, leafSize = leafSize, tau = 0.5, distanceMetric=distanceMetric)
        it("should have correct size") {
          tree.size shouldBe points.size
        }
        it("should return an iterator that goes through all data points") {
          tree.iterator.toIterable should contain theSameElementsAs points
        }
        it("works for every point to identify itself") {
          points.foreach(v => tree.query(v.vector, 1).head._1 shouldBe v)
        }
        it("has consistent size and iterator") {
          def check(tree: Tree): Unit = {
            tree match {
              case t: SpillTree =>
                t.iterator.size shouldBe t.size

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

  describe("HybridTree") {
    val origin = Vectors.dense(0, 0)
    describe("can be constructed with empty data") {
      val tree = HybridTree.build(IndexedSeq.empty[RowWithVector], tau = 0.0, distanceMetric=EuclideanDistanceMetric)
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

    describe("with equidistant points on a circle") {
      val n = 12
      val points = (1 to n).map {
        i => new RowWithVector(Vectors.dense(math.sin(2 * math.Pi * i / n), math.cos(2 * math.Pi * i / n)), null)
      }
      val leafSize = n / 4
      val tree = HybridTree.build(points, leafSize = leafSize, tau = 0.5, distanceMetric=EuclideanDistanceMetric)
      it("should have correct size") {
        tree.size shouldBe points.size
      }
      it("should return an iterator that goes through all data points") {
        tree.iterator.toIterable should contain theSameElementsAs points
      }
    }
  }
}
