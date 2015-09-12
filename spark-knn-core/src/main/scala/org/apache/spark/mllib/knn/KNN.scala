package org.apache.spark.mllib.knn

import org.apache.spark.{Partitioner, Logging}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.{ShuffledRDD, RDD}

import scala.annotation.tailrec
import scala.reflect.ClassTag


class KNN (val topTreeSize: Int,
           val topTreeLeafSize: Int
            ) extends Serializable with Logging {
  def run(data: RDD[Vector]): KNNRDD[hasVector] = {
    run(data.map(hasVector.apply))
  }

  def run[T <: hasVector : ClassTag](data: RDD[T]): KNNRDD[T] = {
    val sampled = data.sample(false, topTreeSize / data.count()).collect()
    val topTree = MetricTree(sampled, topTreeLeafSize)
    val part = new KNNPartitioner(topTree)
    val repartitioned = new ShuffledRDD[VectorWithNorm, T, T](data.map(x => (x.vectorWithNorm, x)), part)
    val trees = repartitioned.mapPartitions{
      itr =>
        val childTree = MetricTree(itr.map(_._2).toIndexedSeq)
        Iterator(childTree)
    }
    new KNNRDD[T](topTree, trees)
  }
}

class KNNPartitioner[T <: hasVector](tree: Tree[T]) extends Partitioner {
  override def numPartitions: Int = tree.leafCount

  override def getPartition(key: Any): Int = {
    key match {
      case v: VectorWithNorm => searchIndex(v)
      case _ => throw new IllegalArgumentException(s"Key must be of type Vector but got: $key")
    }
  }

  @tailrec
  private[this] def searchIndex(v: VectorWithNorm, tree: Tree[T] = tree, acc: Int = 0): Int = {
    tree match {
      case node: MetricNode[T] =>
        val leftDistance = node.leftPivot.fastSquaredDistance(v)
        val rightDistance = node.rightPivot.fastSquaredDistance(v)
        if(leftDistance < rightDistance) {
          searchIndex(v, node.leftChild, acc)
        } else {
          searchIndex(v, node.rightChild, acc + node.leftChild.leafCount)
        }
      case _ => acc
    }
  }
}
