package org.apache.spark.mllib.knn

import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.{Matchers, FunSpec}

class MetricTreeSpec extends FunSpec with Matchers {

  describe("MetricTree") {
    describe("without duplicates") {
      val data = (-5 to 5).flatMap(i => (-5 to 5).map(j => Vectors.dense(i, j)))
      val tree = MetricTree.create(data)
      import tree._
      it("should have correct size") {
        tree.size shouldBe data.size
        (leftChild.size + rightChild.size) shouldBe tree.size
      }
      it("should return an iterator that goes through all data points") {
        tree.iterator.toIndexedSeq.sortWith{
          case (v1, v2) => v1(0) < v2(0) || (v1(0) == v2(0) && v1(1) < v2(1))
        } shouldBe data
      }
      it("should return vector itself for those in input set") {
        data.foreach(v => tree.query(v).head shouldBe v)
      }
      val origin = Vectors.dense(0, 0)
      it("should return nearest neighbors correctly") {
        tree.query(origin, 5).toSet shouldBe Set(
          Vectors.dense(-1, 0),
          Vectors.dense(1, 0),
          Vectors.dense(0, -1),
          Vectors.dense(0, 1),
          Vectors.dense(0, 0)
        )
        tree.query(origin, 9).toSet shouldBe
          (-1 to 1).flatMap(i => (-1 to 1).map(j => Vectors.dense(i, j))).toSet
      }
    }
  }
}
