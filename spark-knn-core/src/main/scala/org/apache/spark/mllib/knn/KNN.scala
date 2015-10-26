package org.apache.spark.mllib.knn

import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Partitioner, Logging}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.{ShuffledRDD, RDD}
import breeze.linalg._
import breeze.stats._

import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


class KNN (val topTreeSize: Int,
           val topTreeLeafSize: Int,
          val subTreeLeafSize: Int,
           val tau: Option[Double] = None
            ) extends Serializable with Logging {
  def run(data: RDD[Vector]): KNNRDD[hasVector] = {
    run(data.map(hasVector.apply))
  }

  def run[T <: hasVector : ClassTag](data: RDD[T]): KNNRDD[T] = {
    val sampled = data.sample(false, topTreeSize / data.count()).collect()
    val topTree = MetricTree.build(sampled, topTreeLeafSize)
    val part = new KNNPartitioner(topTree)
    val repartitioned = new ShuffledRDD[VectorWithNorm, T, T](data.map(x => (x.vectorWithNorm, x)), part)

    val _tau = tau.getOrElse(estimateTau(data))
    logInfo("Tau is: " + _tau)
    val trees = repartitioned.mapPartitions{
      itr =>
        val childTree = HybridTree.build(itr.map(_._2).toIndexedSeq, subTreeLeafSize, tau = _tau)
        Iterator(childTree)
    }.persist(StorageLevel.MEMORY_AND_DISK)

    new KNNRDD[T](topTree, _tau, trees)
  }

  def estimateTau[T <: hasVector](data: RDD[T], sampleSize: Seq[Int] = 100 to 10000 by 50): Double = {
    val total = data.count().toDouble

    val estimators = data.flatMap {
      p =>
        sampleSize.zipWithIndex.filter{ case (size, _) => math.random * total <= size }
          .map{ case (size, index) => (index, p.vectorWithNorm) }
    }
      .groupByKey()
      .map {
      case (index, points) => (points.size, computeAverageDistance(points))
    }.collect().distinct
    logInfo(estimators.toSeq.toString)

    val x = DenseVector(estimators.map{ case(n, _) => math.log(n)})
    val y = DenseVector(estimators.map{ case(_, d) => math.log(d)})

    val xMeanVariance: MeanAndVariance = meanAndVariance(x)
    val xmean = xMeanVariance.mean
    val yMeanVariance: MeanAndVariance = meanAndVariance(y)
    val ymean = yMeanVariance.mean

    val corr = (mean(x :* y) - xmean * ymean) / math.sqrt((mean(x :* x) - xmean * xmean) * (mean(y :* y) - ymean * ymean))

    val beta = corr * yMeanVariance.stdDev / xMeanVariance.stdDev
    val alpha = ymean - beta * xmean
    val rs = math.exp(alpha + beta * math.log(total))

    val d = - 1 / beta
    rs / math.sqrt(d) / 2
  }

  private[this] def computeAverageDistance(points: Iterable[VectorWithNorm]): Double = {
    val distances = points.map(point => points.map(point.fastSquaredDistance).filter(_ > 0).min).map(math.sqrt)
    distances.sum / distances.size
  }
}

/**
 * Partitioner used to map vector to leaf node which determines the partition it goes to
 *
 * @param tree [[MetricTree]] used to find leaf
 */
class KNNPartitioner[T <: hasVector](tree: Tree[T]) extends Partitioner {
  override def numPartitions: Int = tree.leafCount

  override def getPartition(key: Any): Int = {
    key match {
      case v: VectorWithNorm => KNNIndexFinder.searchIndex(v, tree)
      case _ => throw new IllegalArgumentException(s"Key must be of type Vector but got: $key")
    }
  }

}

private[knn] object KNNIndexFinder {
  /**
   * Search leaf index used by KNNPartitioner to partition training points
   *
   * @param v one training point to partition
   * @param tree top tree constructed using sampled points
   * @param acc accumulator used to help determining leaf index
   * @return leaf/partition index
   */
  @tailrec
  def searchIndex[T <: hasVector](v: VectorWithNorm, tree: Tree[T], acc: Int = 0): Int = {
    tree match {
      case node: MetricTree[T] =>
        val leftDistance = node.leftPivot.fastSquaredDistance(v)
        val rightDistance = node.rightPivot.fastSquaredDistance(v)
        if(leftDistance < rightDistance) {
          searchIndex(v, node.leftChild, acc)
        } else {
          searchIndex(v, node.rightChild, acc + node.leftChild.leafCount)
        }
      case _ => acc // reached leaf
    }
  }

  //TODO: Might want to make this tail recursive
  def searchIndecies[T <: hasVector](v: VectorWithNorm, tree: Tree[T], tau: Double, acc: Int = 0): Seq[Int] = {
    tree match {
      case node: MetricTree[T] =>
        val leftDistance = node.leftPivot.fastDistance(v)
        val rightDistance = node.rightPivot.fastDistance(v)

        val buffer = new ArrayBuffer[Int]
        if(leftDistance - rightDistance <= 2 * tau) {
          buffer ++= searchIndecies(v, node.leftChild, tau, acc)
        }

        if (rightDistance - leftDistance <= 2 * tau) {
          buffer ++= searchIndecies(v, node.rightChild, tau, acc + node.leftChild.leafCount)
        }

        buffer
      case _ => Seq(acc) // reached leaf
    }
  }
}
