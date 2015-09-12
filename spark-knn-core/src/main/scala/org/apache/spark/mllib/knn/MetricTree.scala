package org.apache.spark.mllib.knn

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.MLUtils
import breeze.linalg._
import breeze.numerics._

import scala.annotation.tailrec
import scala.collection.mutable
import scala.util.Random

private[knn] abstract class Tree[T <: hasVector] extends Serializable {
  val leftChild: Tree[T]
  val rightChild: Tree[T]
  val size: Int
  val leafCount: Int
  val pivot: VectorWithNorm
  val radius: Double

  def iterator: Iterator[T]

  /**
   * k-NN query using pre-built [[Tree]]
   * @param v vector to query
   * @param k number of nearest neighbor
   * @return a list of neighbor that is nearest to the query vector
   */
  def query(v: Vector, k: Int = 1): Iterable[T] = query(new VectorWithNorm(v), k)
  def query(v: VectorWithNorm, k: Int): Iterable[T] = query(new KNNCandidates[T](v, k)).toIterable

  /**
   * Refine k-NN candidates using data in this [[Tree]]
   */
  private[knn] def query(candidates: KNNCandidates[T]): KNNCandidates[T]

  /**
   * Compute QueryCost defined as || v.center - q || - r
   * when >= v.r node can be pruned
   * for MetricNode this can be used to determine which child does queryVector falls into
   */
  private[knn] def queryCost(candidates: KNNCandidates[T]): Double =
    if(pivot.vector.size > 0) {
      queryCost(candidates.queryVector) - candidates.maxDistance
    } else {
      0.0
    }
  private[knn] def queryCost(v: VectorWithNorm): Double =
    if(pivot.vector.size > 0) {
      pivot.fastDistance(v)
    } else {
      0.0
    }
}

private[knn]
case object Empty extends Tree[Nothing] {
  override val leftChild = this
  override val rightChild = this
  override val size = 0
  override val leafCount = 0
  override val pivot = new VectorWithNorm(Vectors.dense(Array.empty[Double]), 0.0)
  override val radius = 0.0

  override def iterator: Iterator[Nothing] = Iterator.empty
  override def query(candidates: KNNCandidates[Nothing]): KNNCandidates[Nothing] = candidates
  def apply[T <: hasVector]: Tree[T] = this.asInstanceOf[Tree[T]]
}

private[knn]
case class Leaf[T <: hasVector] (data: IndexedSeq[T],
                    pivot: VectorWithNorm,
                    radius: Double) extends Tree[T] {
  override val leftChild = Empty[T]
  override val rightChild = Empty[T]
  override val size = data.size
  override val leafCount = 1

  override def iterator: Iterator[T] = data.iterator

  override def query(candidates: KNNCandidates[T]): KNNCandidates[T] = {
    val sorted = data
      .map{ v => (v, candidates.queryVector.fastDistance(v.vectorWithNorm)) }
      .sortBy(_._2)

    for((v, d) <- sorted if candidates.notFull ||  d < candidates.maxDistance)
      candidates.insert(v)

    candidates
  }
}

private[knn]
object Leaf {
  def apply[T <: hasVector](data: IndexedSeq[T]): Leaf[T] = {
    val vectors = data.map(_.vector.toBreeze)
    val (minV, maxV) = vectors.foldLeft((vectors.head, vectors.head)) {
      case ((accMin, accMax), bv) =>
        (min(accMin, bv), max(accMax, bv))
    }
    val pivot = new VectorWithNorm((minV + maxV) / 2.0)
    val radius = math.sqrt(squaredDistance(minV, maxV)) / 2.0
    Leaf(data, pivot, radius)
  }
}

private[knn]
case class MetricNode[T <: hasVector](leftChild: Tree[T],
                         leftPivot: VectorWithNorm,
                         rightChild: Tree[T],
                         rightPivot: VectorWithNorm,
                         pivot: VectorWithNorm,
                         radius: Double
                          ) extends Tree[T] {
  override val size = leftChild.size + rightChild.size
  override val leafCount = leftChild.leafCount + rightChild.leafCount

  override def iterator: Iterator[T] = leftChild.iterator ++ rightChild.iterator
  override def query(candidates: KNNCandidates[T]): KNNCandidates[T] = {
    lazy val leftQueryCost = leftChild.queryCost(candidates)
    lazy val rightQueryCost = rightChild.queryCost(candidates)
    // only query if at least one of the children is worth looking
    if(candidates.notFull || leftQueryCost < leftChild.radius || rightQueryCost < rightChild.radius ){
      val remainingChild = {
        if (leftQueryCost <= rightQueryCost) {
          leftChild.query(candidates)
          rightChild
        } else {
          rightChild.query(candidates)
          leftChild
        }
      }
      // check again to see if the remaining child is still worth looking
      if (candidates.notFull || remainingChild.queryCost(candidates) < remainingChild.radius) {
        remainingChild.query(candidates)
      }
    }
    candidates
  }
}

object MetricTree {
  /**
   * Build a [[Tree]] that facilitate k-NN query
   *
   * @param data vectors that contain all training data
   * @param rand random number generator used in pivot point selecting
   * @return a [[Tree]] can be used to do k-NN query
   */
  def apply[T <: hasVector](data: IndexedSeq[T], leafSize: Int = 1, rand: Random = new Random): Tree[T] = {
    val size = data.size
    if(size == 0) {
      Empty[T]
    } else if(size <= leafSize) {
      Leaf(data)
    } else {
      val randomPivot = data(rand.nextInt(size)).vectorWithNorm
      val leftPivot = data.maxBy(v => randomPivot.fastSquaredDistance(v.vectorWithNorm)).vectorWithNorm
      if(leftPivot == randomPivot) {
        // all points are identical (including only one point left)
        Leaf(data, randomPivot, 0.0)
      } else {
        val rightPivot = data.maxBy(v => leftPivot.fastSquaredDistance(v.vectorWithNorm)).vectorWithNorm
        val pivot = new VectorWithNorm(Vectors.fromBreeze((leftPivot.vector.toBreeze + rightPivot.vector.toBreeze) / 2.0))
        val radius = data.maxBy(v => pivot.fastSquaredDistance(v.vectorWithNorm)).vectorWithNorm.fastDistance(pivot)
        val (leftPartition, rightPartition) = data.partition{
          v => leftPivot.fastSquaredDistance(v.vectorWithNorm) < rightPivot.fastSquaredDistance(v.vectorWithNorm)
        }

        MetricNode(
          apply(leftPartition, leafSize, rand),
          leftPivot,
          apply(rightPartition, leafSize, rand),
          rightPivot,
          pivot,
          radius
        )
      }
    }
  }
}

//
//case class SpillTree(leftChild: Tree,
//                     rightChild: Tree) extends Tree {
//
//}

private[knn]
class KNNCandidates[T <: hasVector](val queryVector: VectorWithNorm, val k: Int) extends Serializable {
  private[this] var _distance: Double = _
  private[this] val candidates = mutable.PriorityQueue.empty[T] {
    Ordering.by(x => queryVector.fastSquaredDistance(x.vectorWithNorm))
  }

  def maxDistance: Double = _distance
  def insert(v: T*): Unit = {
    while(candidates.size > k - v.size) candidates.dequeue()
    candidates.enqueue(v: _*)
    _distance = candidates.head.vectorWithNorm.fastDistance(queryVector)
  }
  def toIterable: Iterable[T] = candidates
  def notFull: Boolean = candidates.size < k
}
