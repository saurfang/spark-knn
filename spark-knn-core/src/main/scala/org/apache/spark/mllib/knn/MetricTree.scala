package org.apache.spark.mllib.knn

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.MLUtils
import breeze.linalg._
import breeze.numerics._

import scala.annotation.tailrec
import scala.collection.mutable
import scala.util.Random

private[knn] abstract class Tree[T] extends Serializable {
  val leftChild: Tree[T]
  val rightChild: Tree[T]
  val size: Int
  val leafCount: Int
  val pivot: VectorWithNorm
  val radius: Double

  def iterator: Iterator[(Vector, T)]

  /**
   * k-NN query using pre-built [[Tree]]
   * @param v vector to query
   * @param k number of nearest neighbor
   * @return a list of neighbor that is nearest to the query vector
   */
  def query(v: Vector, k: Int = 1): Iterable[(Vector, T)] = query(new VectorWithNorm(v), k)
  def query(v: VectorWithNorm, k: Int): Iterable[(Vector, T)] = query(new KNNCandidates[T](v, k)).toIterable

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

  override def iterator: Iterator[(Vector, Nothing)] = Iterator.empty
  override def query(candidates: KNNCandidates[Nothing]): KNNCandidates[Nothing] = candidates
  def apply[T]: Tree[T] = this.asInstanceOf[Tree[T]]
}

private[knn]
case class Leaf[T] (data: IndexedSeq[(VectorWithNorm, T)],
                    pivot: VectorWithNorm,
                    radius: Double) extends Tree[T] {
  override val leftChild = Empty[T]
  override val rightChild = Empty[T]
  override val size = data.size
  override val leafCount = 1

  override def iterator: Iterator[(Vector, T)] = data.iterator.map{ case (k, v) => (k.vector, v) }

  override def query(candidates: KNNCandidates[T]): KNNCandidates[T] = {
    val sorted = data
      .map{ case (k, v) => ((k, v), candidates.queryVector.fastDistance(k)) }
      .sortBy(_._2)

    for((v, d) <- sorted if candidates.notFull ||  d < candidates.maxDistance)
      candidates.insert(v)

    candidates
  }
}

private[knn]
object Leaf {
  def apply[T](data: IndexedSeq[(VectorWithNorm, T)]): Leaf[T] = {
    val vectors = data.map(_._1.vector.toBreeze)
    val (minV, maxV) = data.foldLeft((vectors.head, vectors.head)) {
      case ((accMin, accMax), v) =>
        val bv = v._1.vector.toBreeze
        (min(accMin, bv), max(accMax, bv))
    }
    val pivot = new VectorWithNorm((minV + maxV) / 2.0)
    val radius = math.sqrt(squaredDistance(minV, maxV)) / 2.0
    Leaf(data, pivot, radius)
  }
}

private[knn]
case class MetricNode[T](leftChild: Tree[T],
                         leftPivot: VectorWithNorm,
                         rightChild: Tree[T],
                         rightPivot: VectorWithNorm,
                         pivot: VectorWithNorm,
                         radius: Double
                          ) extends Tree[T] {
  override val size = leftChild.size + rightChild.size
  override val leafCount = leftChild.leafCount + rightChild.leafCount

  override def iterator: Iterator[(Vector, T)] = leftChild.iterator ++ rightChild.iterator
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
  def create[T](data: IndexedSeq[(Vector, T)], leafSize: Int = 1, rand: Random = new Random): Tree[T] = {
    apply(data.map{ case (k, v) => (new VectorWithNorm(k), v)}, leafSize, rand)
  }

  /**
   * Build a [[Tree]] that facilitate k-NN query
   *
   * @param data vectors that contain all training data
   * @param rand random number generator used in pivot point selecting
   * @return a [[Tree]] can be used to do k-NN query
   */
  def apply[T](data: IndexedSeq[(VectorWithNorm, T)], leafSize: Int, rand: Random): Tree[T] = {
    val size = data.size
    if(size == 0) {
      Empty[T]
    } else if(size <= leafSize) {
      Leaf(data)
    } else {
      val randomPivot = data(rand.nextInt(size))._1
      val leftPivot = data.maxBy(v => randomPivot.fastSquaredDistance(v._1))._1
      if(leftPivot == randomPivot) {
        // all points are identical (including only one point left)
        Leaf(data, randomPivot, 0.0)
      } else {
        val rightPivot = data.maxBy(v => leftPivot.fastSquaredDistance(v._1))._1
        val pivot = new VectorWithNorm(Vectors.fromBreeze((leftPivot.vector.toBreeze + rightPivot.vector.toBreeze) / 2.0))
        val radius = data.maxBy(v => pivot.fastSquaredDistance(v._1))._1.fastDistance(pivot)
        val (leftPartition, rightPartition) = data.partition{
          v => leftPivot.fastSquaredDistance(v._1) < rightPivot.fastSquaredDistance(v._1)
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
class VectorWithNorm(val vector: Vector, val norm: Double) extends Serializable {

  def this(vector: Vector) = this(vector, Vectors.norm(vector, 2.0))
  def this(vector: breeze.linalg.Vector[Double]) = this(Vectors.fromBreeze(vector))

  def fastSquaredDistance(v: VectorWithNorm): Double = {
    MLUtils.fastSquaredDistance(vector, norm, v.vector, v.norm)
  }
  def fastDistance(v: VectorWithNorm): Double = math.sqrt(fastSquaredDistance(v))

  override def toString: String = s"$vector ($norm)"
}

private[knn]
class KNNCandidates[T](val queryVector: VectorWithNorm, val k: Int) extends Serializable {
  private[this] var _distance: Double = _
  private[this] val candidates = mutable.PriorityQueue.empty[(VectorWithNorm, T)] {
    Ordering.by(x => queryVector.fastSquaredDistance(x._1))
  }

  def maxDistance: Double = _distance
  def insert(v: (VectorWithNorm, T)*): Unit = {
    while(candidates.size > k - v.size) candidates.dequeue()
    candidates.enqueue(v: _*)
    _distance = candidates.head._1.fastDistance(queryVector)
  }
  def toIterable: Iterable[(Vector, T)] = candidates.map{case (key, value) => (key.vector, value) }
  def notFull: Boolean = candidates.size < k
}
