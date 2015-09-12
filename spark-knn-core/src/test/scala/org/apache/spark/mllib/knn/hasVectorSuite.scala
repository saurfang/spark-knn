package org.apache.spark.mllib.knn

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.scalatest.{Matchers, FunSuite}

class hasVectorSuite extends FunSuite with Matchers {
  private[this] val origin = Vectors.dense(0.0)

  test("hasVector works for vector") {
    val wrapped = hasVector(origin)
    wrapped.vector shouldBe origin
    wrapped.vectorWithNorm.vector shouldBe origin
    wrapped.vectorWithNorm.norm shouldBe 0.0
  }

  test("hasVector works for pair") {
    val wrapped = hasVector(origin, "test")
    wrapped.vector shouldBe origin
    wrapped._2 shouldBe "test"
  }

  test("hasVector works for tuple") {
    val wrapped = hasVector((origin, "test"))
    wrapped.vector shouldBe origin
    wrapped._2 shouldBe "test"
  }

  test("hasVector works for LabeledPoint") {
    val wrapped = hasVector(new LabeledPoint(1.0, origin))
    wrapped.vector shouldBe origin
    wrapped.label shouldBe 1.0
  }
}
