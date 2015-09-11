package org.apache.spark.mllib.knn

import org.apache.spark.{Partitioner, Logging}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.{ShuffledRDD, RDD}

import scala.annotation.tailrec


class KNN (val topTreeSize: Int,
           val topTreeLeafSize: Int
            ) extends Serializable with Logging {
  def run[T](data: RDD[(Vector, T)]): KNNRDD[T] = {
    val sampled = data.sample(false, topTreeSize / data.count()).collect()
    val topTree = MetricTree.create(sampled, topTreeLeafSize)
    val part = new KNNPartitioner(topTree)
    val repartitioned = new ShuffledRDD[Vector, T, T](data, part)
    val trees = repartitioned.mapPartitions{
      itr =>
        val childTree = MetricTree.create(itr.toIndexedSeq)
        Iterator(childTree)
    }
    new KNNRDD[T](topTree, trees)
  }
}

class KNNPartitioner[T](tree: Tree[T]) extends Partitioner {
  override def numPartitions: Int = tree.leafCount

  override def getPartition(key: Any): Int = {
    key match {
      case v: Vector => searchIndex(new VectorWithNorm(v))
      case _ => throw new IllegalArgumentException(s"Key must be of type Vector but got: $key")
    }
  }

  @tailrec
  private[this] def searchIndex(v: VectorWithNorm, tree: Tree[T] = tree, acc: Int = 0): Int = {
    tree match {
      case node: MetricNode[T] =>
        val leftDistance = node.leftPivot.fastSquaredDistance(v)
        val rightDistance = node.leftPivot.fastSquaredDistance(v)
        if(leftDistance < rightDistance) {
          searchIndex(v, node.leftChild, acc)
        } else {
          searchIndex(v, node.rightChild, acc + node.leftChild.leafCount)
        }
      case _ => acc
    }
  }
}