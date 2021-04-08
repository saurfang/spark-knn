package org.apache.spark.ml.knn

import org.apache.spark.ml.knn.KNN.VectorWithNorm
import org.apache.spark.mllib.knn.KNNUtils


object DistanceMetric {
  def apply(metric: String): DistanceMetric = {
    metric match {
      case "" | "euclidean" => EuclideanDistanceMetric
      case "nan_euclidean" => NaNEuclideanDistanceMetric
      case _ => throw new IllegalArgumentException(s"Unsupported distance metric: $metric")
    }
  }
}
trait DistanceMetric {
  def fastSquaredDistance(v1: VectorWithNorm, v2: VectorWithNorm): Double

  def fastDistance(v1: VectorWithNorm, v2: VectorWithNorm): Double = {
    math.sqrt(fastSquaredDistance(v1, v2))
  }
}

object EuclideanDistanceMetric extends DistanceMetric with Serializable {
  override def fastSquaredDistance(v1: VectorWithNorm, v2: VectorWithNorm): Double = {
    KNNUtils.fastSquaredDistance(v1.vector, v1.norm, v2.vector, v2.norm)
  }
}

/**
 * Calculate NaN-Euclidean distance by using only non NaN values in each vector
 */
object NaNEuclideanDistanceMetric extends DistanceMetric with Serializable {

  class InfiniteZeroIterator(step: Int = 1) extends Iterator[(Int, Double)] {
    var index = -1 * step
    override def hasNext: Boolean = true
    override def next(): (Int, Double) = {
      index += step
      (index, 0.0)
    }
  }
  override def fastSquaredDistance(v1: VectorWithNorm, v2: VectorWithNorm): Double = {
    var it1 = v1.vector.activeIterator
    var it2 = v2.vector.activeIterator
    if(!it1.hasNext && !it2.hasNext) return 0.0
    if(!it1.hasNext) {
      it1 = new InfiniteZeroIterator
    } else if(!it2.hasNext) {
      it2 = new InfiniteZeroIterator
    }
    var result = 0.0
    // initial case
    var (idx1, val1) = it1.next()
    var (idx2, val2) = it2.next()
    // iterator over the vectors
    while((it1.hasNext || it2.hasNext) && !(it1.isInstanceOf[InfiniteZeroIterator] && it2.isInstanceOf[InfiniteZeroIterator])) {
      var (advance1, advance2) = (false, false)
      val (left, right) = if(idx1 < idx2) {
        // advance iterator on first vector
        advance1 = true
        (val1, 0.0)
      } else if(idx1 > idx2) {
        // advance iterator on second vector
        advance2 = true
        (0.0, val2)
      } else {
        // indexes matches
        advance1 = true
        advance2 = true
        (val1, val2)
      }
      if(!left.isNaN && !right.isNaN) {
        result += Math.pow(left - right, 2)
      }
      if(advance1) {
        if(!it1.hasNext) it1 = new InfiniteZeroIterator
        val next1 = it1.next()
        idx1 = next1._1
        val1 = next1._2
      }
      if(advance2) {
        if(!it2.hasNext) it2 = new InfiniteZeroIterator
        val next2 = it2.next()
        idx2 = next2._1
        val2 = next2._2
      }
    }
    if(idx1 == idx2 && !val1.isNaN && !val2.isNaN) {
      result += Math.pow(val1 - val2, 2)
    }
    result
  }
}