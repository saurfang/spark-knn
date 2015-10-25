package org.apache.spark.mllib.knn

import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.{Matchers, FunSpec}

class SpillTreeSpec extends FunSpec with Matchers {
  describe("SpillTree") {
    val origin = Vectors.dense(0, 0)
    describe("can be constructed with empty data") {
      val tree = SpillTree.build(IndexedSeq.empty[hasVector], tau = 0.0)
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

    describe("with equidistant points on a circle") {
      val n = 12
      val points = (1 to n).map {
        i => hasVector(Vectors.dense(math.sin(2 * math.Pi * i / n), math.cos(2 * math.Pi * i / n)))
      }
      val leafSize = n / 4
      describe("built with tau = 0.0") {
        val tree = SpillTree.build(points, leafSize = leafSize, tau = 0.0)
        it("should have correct size") {
          tree.size shouldBe points.size
        }
        it("should return an iterator that goes through all data points") {
          tree.iterator.toIterable should contain theSameElementsAs points
        }
        it("works for every point to identify itself") {
          points.foreach(v => tree.query(v.vector).head shouldBe v)
        }
      }
      describe("built with tau = 0.4") {
        val tree = SpillTree.build(points, leafSize = leafSize , tau = 0.4)
        it("should have correct size") {
          tree.size shouldBe points.size
        }
        it("should return an iterator that goes through all data points") {
          tree.iterator.toIterable should contain theSameElementsAs points
        }
        it("has consistent size and iterator") {
          def check(tree: Tree[hasVector]): Unit = {
            tree match {
              case t: SpillTree[hasVector] =>
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
      val tree = HybridTree.build(IndexedSeq.empty[hasVector], tau = 0.0)
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

    describe("with equidistant points on a circle") {
      val n = 12
      val points = (1 to n).map {
        i => hasVector(Vectors.dense(math.sin(2 * math.Pi * i / n), math.cos(2 * math.Pi * i / n)))
      }
      val leafSize = n / 4
      val tree = HybridTree.build(points, leafSize = leafSize, tau = 0.4)
      it("should have correct size") {
        tree.size shouldBe points.size
      }
      it("should return an iterator that goes through all data points") {
        tree.iterator.toIterable should contain theSameElementsAs points
      }
      it("has consistent size and iterator") {
        def check(tree: Tree[hasVector]): Unit = {
          tree match {
            case t: SpillTree[hasVector] =>
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
