package org.apache.spark.ml.knn

import org.apache.spark.ml.knn.KNN.{RowWithVector, VectorWithNorm}
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

class DistanceMetricSpec extends AnyFunSpec with Matchers {

  describe("EuclideanDistanceMetric") {
    val distanceMetric = EuclideanDistanceMetric
    describe("calculate distance between two dense vectors") {
      val v1 = new VectorWithNorm(Vectors.dense(1, 1))
      val v2 = new VectorWithNorm(Vectors.dense(-1, -1))

      it("should return distance for vector and self") {
        distanceMetric.fastDistance(v1, v1) shouldBe 0.0
      }
      it("should return distance for two vectors") {
        distanceMetric.fastDistance(v1, v2) shouldBe Math.sqrt(8.0)
      }
    }
  }

  describe("NaNEuclideanDistanceMetric") {
    val distanceMetric = NaNEuclideanDistanceMetric
    describe("calculate distance between two dense vectors with valid values") {
      val v1 = new VectorWithNorm(Vectors.dense(1, 1))
      val v2 = new VectorWithNorm(Vectors.dense(-1, -1))

      it("should return distance for vector and self") {
        distanceMetric.fastDistance(v1, v1) shouldBe 0.0
      }
      it("should return distance for two vectors") {
        distanceMetric.fastDistance(v1, v2) shouldBe Math.sqrt(8.0)
      }
    }
    describe("calculate distance between two dense vectors with invalid values") {
      val v1 = new VectorWithNorm(Vectors.dense(1, 1, Double.NaN, Double.NaN, 1))
      val v2 = new VectorWithNorm(Vectors.dense(-1, Double.NaN, -1, -1, -1))

      it("should return distance for vector and self") {
        distanceMetric.fastDistance(v1, v1) shouldBe 0.0
      }
      it("should return distance for two vectors") {
        distanceMetric.fastDistance(v1, v2) shouldBe Math.sqrt(8.0)
      }
    }
    describe("calculate distance between two sparse vectors with invalid values") {
      val v1 = new VectorWithNorm(Vectors.sparse(5, Seq((1, 1.0), (2, Double.NaN), (3, Double.NaN), (4, 1.0))))
      val v2 = new VectorWithNorm(Vectors.sparse(5, Seq((0, -1.0), (1, Double.NaN), (4, -1.0))))

      it("should return distance for vector and self") {
        distanceMetric.fastDistance(v1, v1) shouldBe 0.0
      }
      it("should return distance for two vectors") {
        distanceMetric.fastDistance(v1, v2) shouldBe Math.sqrt(5.0)
      }
    }
    describe("calculate distance between a sparse vectors and all 0 vector") {
      val v1 = new VectorWithNorm(Vectors.sparse(5, Seq((0, 0.0), (1, 0.0), (3, 0.0), (4, 0.0))))
      val v2 = new VectorWithNorm(Vectors.sparse(5, Seq((0, -1.0), (1, Double.NaN), (4, -1.0))))

      it("should return distance for vector and self") {
        distanceMetric.fastDistance(v1, v1) shouldBe 0.0
      }
      it("should return distance for two vectors") {
        distanceMetric.fastDistance(v1, v2) shouldBe Math.sqrt(2.0)
      }
    }
    describe("calculate distance between a sparse vectors and empty vector") {
      val v1 = new VectorWithNorm(Vectors.sparse(5, Seq()))
      val v2 = new VectorWithNorm(Vectors.sparse(5, Seq((0, -1.0), (1, Double.NaN), (4, -1.0))))

      it("should return distance for vector and self") {
        distanceMetric.fastDistance(v1, v1) shouldBe 0.0
      }
      it("should return distance for two vectors") {
        distanceMetric.fastDistance(v1, v2) shouldBe Math.sqrt(2.0)
      }
    }
  }
}
