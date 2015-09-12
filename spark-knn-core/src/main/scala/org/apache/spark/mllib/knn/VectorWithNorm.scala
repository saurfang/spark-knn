package org.apache.spark.mllib.knn

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.util.MLUtils

/**
 * Provides common interface to access [[Vector]] and [[VectorWithNorm]]
 */
trait hasVector extends Serializable {
  def vector: Vector

  @transient lazy private[knn] val vectorWithNorm: VectorWithNorm = new VectorWithNorm(vector)
}

/**
 * Helper functions to mixin [[hasVector]] trait into common classes
 */
object hasVector {
  def apply(v: Vector): hasVector = new hasVector {
    override def vector: Vector = v
  }

  def apply[T](v: Vector, o: T): (Vector, T) with hasVector =
    new Tuple2(v, o) with hasVector {
      override val vector: Vector = v
    }

  def apply[T](v: (Vector, T)): (Vector, T) with hasVector = apply(v._1, v._2)

  def apply(lp: LabeledPoint): LabeledPoint with hasVector =
    new LabeledPoint(lp.label, lp.features) with hasVector {
      override val vector: Vector = lp.features
    }
}

private[knn]
class VectorWithNorm(val vector: Vector, val norm: Double) extends Serializable {

  def this(vector: Vector) = this(vector, Vectors.norm(vector, 2.0))
  def this(vector: breeze.linalg.Vector[Double]) = this(Vectors.fromBreeze(vector))

  def fastSquaredDistance(v: VectorWithNorm): Double = {
    MLUtils.fastSquaredDistance(vector, norm, v.vector, v.norm)
  }
  def fastDistance(v: VectorWithNorm): Double = math.sqrt(fastSquaredDistance(v))

  override def toString: String = s"$vector ($norm)"
}
