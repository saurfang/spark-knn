package org.apache.spark.mllib.knn

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.MLUtils
import breeze.linalg._
import breeze.numerics._

import scala.collection.mutable
import scala.util.Random

private[knn] abstract class Tree extends Serializable {
  val leftChild: Tree
  val rightChild: Tree
  val size: Int
  val pivot: VectorWithNorm
  val radius: Double

  def iterator: Iterator[Vector]

  /**
   * k-NN query using pre-built [[Tree]]
   * @param v vector to query
   * @param k number of nearest neighbor
   * @return a list of neighbor that is nearest to the query vector
   */
  def query(v: Vector, k: Int = 1): Iterable[Vector] = query(new VectorWithNorm(v), k)
  def query(v: VectorWithNorm, k: Int): Iterable[Vector] = query(new KNNCandidates(v, k)).toIterable

  /**
   * Refine k-NN candidates using data in this [[Tree]]
   */
  private[knn] def query(candidates: KNNCandidates): KNNCandidates

  /**
   * Compute QueryCost defined as || v.center - q || - r
   * when >= v.r node can be pruned
   * for MetricNode this can be used to determine which child does queryVector falls into
   */
  private[knn] def queryCost(candidates: KNNCandidates): Double =
    if(pivot.vector.size > 0)
      pivot.fastDistance(candidates.queryVector) - candidates.maxDistance
    else
      0.0
}

case object Empty extends Tree {
  val leftChild = this
  val rightChild = this
  val size = 0
  val pivot = new VectorWithNorm(Vectors.dense(Array.empty[Double]), 0.0)
  val radius = 0.0

  override def iterator: Iterator[Vector] = Iterator.empty
  override def query(candidates: KNNCandidates): KNNCandidates = candidates
}

case class Leaf(pivot: VectorWithNorm, size: Int) extends Tree {
  val leftChild = Empty
  val rightChild = Empty
  val radius = 0.0

  override def iterator: Iterator[Vector] = Iterator.fill(size)(pivot.vector)

  override def query(candidates: KNNCandidates): KNNCandidates = {
    var count = 0
    val distance = pivot.fastDistance(candidates.queryVector)
    // add as many as we can until candidates are full or our distance isn't better than worst distance
    while(count < size && (candidates.notFull || distance < candidates.maxDistance)) {
      candidates.insert(pivot)
      count += 1
    }
    candidates
  }
}

case class MetricNode(leftChild: Tree,
                      rightChild: Tree,
                      pivot: VectorWithNorm,
                      radius: Double,
                      size: Int
                       ) extends Tree {
  override def iterator: Iterator[Vector] = leftChild.iterator ++ rightChild.iterator
  override def query(candidates: KNNCandidates): KNNCandidates = {
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
      if(candidates.notFull || remainingChild.queryCost(candidates) < remainingChild.radius)
        remainingChild.query(candidates)
    }
    candidates
  }
}

object MetricTree {
  def create(data: IndexedSeq[Vector], rand: Random = new Random): Tree = {
    apply(data.map(x => new VectorWithNorm(x)), rand)
  }

  /**
   * Build a [[Tree]] that facilitate k-NN query
   *
   * @param data vectors that contain all training data
   * @param rand random number generator used in pivot point selecting
   * @return a [[Tree]] can be used to do k-NN query
   */
  def apply(data: IndexedSeq[VectorWithNorm], rand: Random): Tree = {
    val size = data.size
    if(size == 0) {
      Empty
    } else {
      val randomPivot = data(rand.nextInt(size))
      val leftPivot = data.maxBy(randomPivot.fastSquaredDistance)
      if(leftPivot == randomPivot) {
        // all points are identical (including only one point left)
        Leaf(leftPivot, data.size)
      } else {
        val rightPivot = data.maxBy(leftPivot.fastSquaredDistance)
        val pivot = new VectorWithNorm(Vectors.fromBreeze((leftPivot.vector.toBreeze + rightPivot.vector.toBreeze) / 2.0))
        val radius = data.maxBy(pivot.fastSquaredDistance).fastDistance(pivot)
        val (leftPartition, rightPartition) = data.partition(v => leftPivot.fastSquaredDistance(v) < rightPivot.fastSquaredDistance(v))

        MetricNode(apply(leftPartition, rand), apply(rightPartition, rand), pivot, radius, size)
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

  def this(array: Array[Double]) = this(Vectors.dense(array))

  /** Converts the vector to a dense vector. */
  def toDense: VectorWithNorm = new VectorWithNorm(Vectors.dense(vector.toArray), norm)

  def fastSquaredDistance(v: VectorWithNorm): Double = {
    MLUtils.fastSquaredDistance(vector, norm, v.vector, v.norm)
  }
  def fastDistance(v: VectorWithNorm): Double = math.sqrt(fastSquaredDistance(v))

  override def toString = s"$vector ($norm)"
}

private[knn]
class KNNCandidates(val queryVector: VectorWithNorm, val k: Int) extends Serializable {
  private[this] var _distance: Double = _
  private[this] val candidates = mutable.PriorityQueue.empty[VectorWithNorm](Ordering.by(queryVector.fastSquaredDistance))

  def maxDistance: Double = _distance
  def insert(v: VectorWithNorm*): Unit = {
    while(candidates.size > k - v.size) candidates.dequeue()
    candidates.enqueue(v: _*)
    _distance = candidates.head.fastDistance(queryVector)
  }
  def toIterable: Iterable[Vector] = candidates.map(_.vector)
  def notFull: Boolean = candidates.size < k
}