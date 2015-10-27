package org.apache.spark.mllib.knn

import breeze.util.TopK
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.{HashPartitioner, Partition, TaskContext}

import scala.reflect.ClassTag

/**
 * `KNNRDD[T]` extends `RDD[T]` by storing objects in a k-NN search tree in each partition.
 * It additionally store a `rootTree` which dictates the partitioning of input points as well
 * as guides the partitioning of search query.
 *
 * @param rootTree top [[Tree]] used to partition input/search vectors
 * @param childrenTree [[RDD]] of [[Tree]]s that facilitates k-NN search in each partition
 * @tparam T type of the data which must implement [[hasVector]] trait so [[Vector]] can be
 *           accessed in k-NN search while additional data may be tied with each returned
 *           neighbor.
 */
class KNNRDD[T <: hasVector : ClassTag] private[knn]
                (val rootTree: Tree[T],
                 val tau: Double,
                @transient childrenTree: RDD[Tree[T]]) extends RDD[T](childrenTree) {

  @DeveloperApi
  override def compute(split: Partition, context: TaskContext): Iterator[T] =
    firstParent[Tree[T]].iterator(split, context).flatMap(_.iterator)

  override protected def getPartitions: Array[Partition] = firstParent.partitions

  override def getPreferredLocations(split: Partition): Seq[String] =
    firstParent.preferredLocations(split)

  def query(data: T, k: Int): Iterable[T] = {
    query(context.parallelize(Seq(data)), k).first()._2
  }

  def query(data: RDD[T], k: Int = 1): RDD[(T, Iterable[T])] = {
    // map each point to a (index, point) pair and repartition
    val searchData = data.zipWithIndex().flatMap {
      point =>
        KNNIndexFinder.searchIndecies(point._1.vectorWithNorm, rootTree, tau).map(i => (i, point))
    }.partitionBy(new HashPartitioner(partitions.length))

    // for each partition, search points within corresponding child tree
    val results = searchData.zipPartitions(childrenTree) {
      (childData, trees) =>
        val tree = trees.next()
        assert(!trees.hasNext)
        childData.map {
          case (_, (point, i)) =>
            val result = tree.query(point.vectorWithNorm, k).map {
              neighbor => (neighbor, neighbor.vectorWithNorm.fastSquaredDistance(point.vectorWithNorm))
            }.toSeq
            (i, (point, result))
        }
    }

    // merge results by point index together and keep topK results
    results.reduceByKey {
      case ((p1, c1), (p2, c2)) => (p1, merge(c1, c2, k))
    }
      .sortByKey()
      .map {
      case (i, (p, c)) => (p, c.map(_._1))
    }
  }

  // note: topK is not serializable
  private[this] def merge(s1: Seq[(T, Double)],
                          s2: Seq[(T, Double)],
                          k: Int): Seq[(T, Double)] = {
    val topK = new TopK[(T, Double)](k)(Ordering.by(- _._2))
    s1.foreach(topK.+=)
    s2.foreach(topK.+=)
    topK.toArray.toSeq
  }
}
