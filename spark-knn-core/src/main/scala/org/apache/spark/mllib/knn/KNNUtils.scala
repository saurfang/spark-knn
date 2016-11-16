package org.apache.spark.mllib.knn

import org.apache.spark.ml.linalg.BLAS._
import org.apache.spark.ml.linalg._

object KNNUtils {
  def fastSquaredDistance(
                           v1: Vector,
                           norm1: Double,
                           v2: Vector,
                           norm2: Double,
                           precision: Double = 1e-6): Double =
    mllibMLUtilsfastSquaredDistance(v1, norm1, v2, norm2, precision)


  // following two functions are copied to convert from mllib to ml
  // @see org.apache.spark.mllib.util.MLUtils


  lazy val EPSILON = {
    var eps = 1.0
    while ((1.0 + (eps / 2.0)) != 1.0) {
      eps /= 2.0
    }
    eps
  }

  /**
    * Returns the squared Euclidean distance between two vectors. The following formula will be used
    * if it does not introduce too much numerical error:
    * <pre>
    *   \|a - b\|_2^2 = \|a\|_2^2 + \|b\|_2^2 - 2 a^T b.
    * </pre>
    * When both vector norms are given, this is faster than computing the squared distance directly,
    * especially when one of the vectors is a sparse vector.
    * @param v1 the first vector
    * @param norm1 the norm of the first vector, non-negative
    * @param v2 the second vector
    * @param norm2 the norm of the second vector, non-negative
    * @param precision desired relative precision for the squared distance
    * @return squared distance between v1 and v2 within the specified precision
    */
  def mllibMLUtilsfastSquaredDistance(
                                          v1: Vector,
                                          norm1: Double,
                                          v2: Vector,
                                          norm2: Double,
                                          precision: Double = 1e-6): Double = {
    val n = v1.size
    require(v2.size == n)
    require(norm1 >= 0.0 && norm2 >= 0.0)
    val sumSquaredNorm = norm1 * norm1 + norm2 * norm2
    val normDiff = norm1 - norm2
    var sqDist = 0.0
    /*
     * The relative error is
     * <pre>
     * EPSILON * ( \|a\|_2^2 + \|b\\_2^2 + 2 |a^T b|) / ( \|a - b\|_2^2 ),
     * </pre>
     * which is bounded by
     * <pre>
     * 2.0 * EPSILON * ( \|a\|_2^2 + \|b\|_2^2 ) / ( (\|a\|_2 - \|b\|_2)^2 ).
     * </pre>
     * The bound doesn't need the inner product, so we can use it as a sufficient condition to
     * check quickly whether the inner product approach is accurate.
     */
    val precisionBound1 = 2.0 * EPSILON * sumSquaredNorm / (normDiff * normDiff + EPSILON)
    if (precisionBound1 < precision) {
      sqDist = sumSquaredNorm - 2.0 * dot(v1, v2)
    } else if (v1.isInstanceOf[SparseVector] || v2.isInstanceOf[SparseVector]) {
      val dotValue = dot(v1, v2)
      sqDist = math.max(sumSquaredNorm - 2.0 * dotValue, 0.0)
      val precisionBound2 = EPSILON * (sumSquaredNorm + 2.0 * math.abs(dotValue)) /
        (sqDist + EPSILON)
      if (precisionBound2 > precision) {
        sqDist = Vectors.sqdist(v1, v2)
      }
    } else {
      sqDist = Vectors.sqdist(v1, v2)
    }
    sqDist
  }

}
