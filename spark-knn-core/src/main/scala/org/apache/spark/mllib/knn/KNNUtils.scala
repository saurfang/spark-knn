package org.apache.spark.mllib.knn

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.util.MLUtils

object KNNUtils {
  def fastSquaredDistance(
                           v1: Vector,
                           norm1: Double,
                           v2: Vector,
                           norm2: Double,
                           precision: Double = 1e-6): Double =
    MLUtils.fastSquaredDistance(v1, norm1, v2, norm2, precision)
}
