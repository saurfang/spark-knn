package org.apache.spark.mllib.knn

import breeze.linalg._
import org.apache.spark.mllib.linalg.{Vector, Vectors}

import scala.collection.mutable
import scala.util.Random

/**
 * A [[Tree]] is used to store data points used in k-NN search. It represents
 * a binary tree node. It keeps track of the pivot vector which closely approximate
 * the center of all vectors within the node. All vectors are within the radius of
 * distance to the pivot vector. Finally it knows the number of leaves to help
 * determining partition index.
 */
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
  // helps to work with the type system so we effectively has Empty[T] without creating more objects
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

  // brute force k-NN search at the leaf
  override def query(candidates: KNNCandidates[T]): KNNCandidates[T] = {
    val sorted = data
      .map{ v => (v, candidates.queryVector.fastDistance(v.vectorWithNorm)) }
      .sortBy(_._2)

    for((v, d) <- sorted if candidates.notFull ||  d < candidates.maxDistance)
      candidates.insert(v, d)

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

/**
 * A [[MetricTree]] represents a MetricNode where data are split into two partitions: left and right.
 * There exists two pivot vectors: leftPivot and rightPivot to determine the partitioning.
 * Pivot vector should be the middle of leftPivot and rightPivot vectors.
 * Points that is closer to leftPivot than to rightPivot belongs to leftChild and rightChild otherwise.
 *
 * During search, because we have information about each child's pivot and radius, we can see if the
 * hyper-sphere intersects with current candidates sphere. If so, we search the child that has the
 * most potential (i.e. the child which has the closest pivot).
 * Once that child has been fully searched, we backtrack to the remaining child and search if necessary.
 *
 * This is much more efficient than naive brute force search. However backtracking can take a lot of time
 * when the number of dimension is high (due to longer time to compute distance and the volume growing much
 * faster than radius).
 */
private[knn]
case class MetricTree[T <: hasVector](leftChild: Tree[T],
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
   * Build a (metric)[[Tree]] that facilitate k-NN query
   *
   * @param data vectors that contain all training data
   * @param rand random number generator used in pivot point selecting
   * @return a [[Tree]] can be used to do k-NN query
   */
  def build[T <: hasVector](data: IndexedSeq[T], leafSize: Int = 1, rand: Random = new Random): Tree[T] = {
    val size = data.size
    if(size == 0) {
      Empty[T]
    } else if(size <= leafSize) {
      Leaf(data)
    } else {
      val randomPivot = data(rand.nextInt(size)).vectorWithNorm
      val leftPivot = data.maxBy(v => randomPivot.fastSquaredDistance(v.vectorWithNorm)).vectorWithNorm
      if(leftPivot == randomPivot) {
        // all points are identical (or only one point left)
        Leaf(data, randomPivot, 0.0)
      } else {
        val rightPivot = data.maxBy(v => leftPivot.fastSquaredDistance(v.vectorWithNorm)).vectorWithNorm
        val pivot = new VectorWithNorm(Vectors.fromBreeze((leftPivot.vector.toBreeze + rightPivot.vector.toBreeze) / 2.0))
        val radius = math.sqrt(data.map(v => pivot.fastSquaredDistance(v.vectorWithNorm)).max)
        val (leftPartition, rightPartition) = data.partition{
          v => leftPivot.fastSquaredDistance(v.vectorWithNorm) < rightPivot.fastSquaredDistance(v.vectorWithNorm)
        }

        MetricTree(
          build(leftPartition, leafSize, rand),
          leftPivot,
          build(rightPartition, leafSize, rand),
          rightPivot,
          pivot,
          radius
        )
      }
    }
  }
}

/**
 * A [[SpillTree]] represents a SpillNode. Just like [[MetricTree]], it splits data into two partitions.
 * However, instead of partition data into exactly two halves, it contains a buffer zone with size of tau.
 * Left child contains all data left to the center plane + tau (in the leftPivot -> rightPivot direction).
 * Right child contains all data right to the center plane - tau.
 *
 * Search doesn't do backtracking but rather adopt a defeatist search where it search the most prominent
 * child and that child only. The buffer ensures such strategy doesn't result in a poor outcome.
 */
private[knn]
case class SpillTree[T <: hasVector](leftChild: Tree[T],
                                      leftPivot: VectorWithNorm,
                                      rightChild: Tree[T],
                                      rightPivot: VectorWithNorm,
                                      pivot: VectorWithNorm,
                                      radius: Double,
                                      tau: Double,
                                      bufferSize: Int
                                       ) extends Tree[T] {
  override val size = leftChild.size + rightChild.size - bufferSize
  override val leafCount = leftChild.leafCount + rightChild.leafCount

  override def iterator: Iterator[T] =
    leftChild.iterator ++ rightChild.iterator.filter(childFilter(leftPivot, rightPivot))

  override def query(candidates: KNNCandidates[T]): KNNCandidates[T] = {
    // because of defeatist search, if we are queried then candidates must be empty
    if (size <= candidates.k) {
      iterator.foreach(candidates.insert)
    } else {
      val leftQueryCost = leftChild.queryCost(candidates)
      val rightQueryCost = rightChild.queryCost(candidates)

      (if (leftQueryCost <= rightQueryCost) leftChild else rightChild).query(candidates)

      // fill candidates with points from other child excluding buffer so we don't double count.
      // depending on K and how high we are in the tree, this can be very expensive and undesirable
      // TODO: revisit this idea when we do large scale testing
      if(candidates.notFull) {
        (if (leftQueryCost <= rightQueryCost) {
          rightChild.iterator.filter(childFilter(leftPivot, rightPivot))
        } else {
          leftChild.iterator.filter(childFilter(rightPivot, leftPivot))
        }).foreach(candidates.tryInsert)
      }
    }
    candidates
  }

  private[this] val childFilter: (VectorWithNorm, VectorWithNorm) => T => Boolean =
    (p1, p2) => p => p.vectorWithNorm.fastDistance(p1) - p.vectorWithNorm.fastDistance(p2) > 2 * tau
}


object SpillTree {
  /**
   * Build a (spill)[[Tree]] that facilitate k-NN query
   *
   * @param data vectors that contain all training data
   * @param rand random number generator used in pivot point selecting
   * @param tau overlapping size
   * @return a [[Tree]] can be used to do k-NN query
   */
  def build[T <: hasVector](data: IndexedSeq[T], leafSize: Int = 1, rand: Random = new Random, tau: Double): Tree[T] = {
    val size = data.size
    if (size == 0) {
      Empty[T]
    } else if (size <= leafSize) {
      Leaf(data)
    } else {
      val randomPivot = data(rand.nextInt(size)).vectorWithNorm
      val leftPivot = data.maxBy(v => randomPivot.fastSquaredDistance(v.vectorWithNorm)).vectorWithNorm
      if (leftPivot == randomPivot) {
        // all points are identical (or only one point left)
        Leaf(data, randomPivot, 0.0)
      } else {
        val rightPivot = data.maxBy(v => leftPivot.fastSquaredDistance(v.vectorWithNorm)).vectorWithNorm
        val pivot = new VectorWithNorm(Vectors.fromBreeze((leftPivot.vector.toBreeze + rightPivot.vector.toBreeze) / 2.0))
        val radius = math.sqrt(data.map(v => pivot.fastSquaredDistance(v.vectorWithNorm)).max)
        val dataWithDistance = data.map(v =>
          (v, leftPivot.fastDistance(v.vectorWithNorm), rightPivot.fastDistance(v.vectorWithNorm))
        )
        val leftPartition = dataWithDistance.filter { case (_, left, right) => left - right <= 2 * tau }.map(_._1)
        val rightPartition = dataWithDistance.filter { case (_, left, right) => right - left <= 2 * tau }.map(_._1)

        SpillTree(
          build(leftPartition, leafSize, rand, tau),
          leftPivot,
          build(rightPartition, leafSize, rand, tau),
          rightPivot,
          pivot,
          radius,
          tau,
          leftPartition.size + rightPartition.size - size
        )
      }
    }
  }
}

object HybridTree {
  /**
   * Build a (hybrid-spill)[[Tree]] that facilitate k-NN query
   *
   * @param data vectors that contain all training data
   * @param rand random number generator used in pivot point selecting
   * @param tau overlapping size
   * @param rho balance threshold
   * @return a [[Tree]] can be used to do k-NN query
   */
  def build[T <: hasVector](data: IndexedSeq[T],
                            leafSize: Int = 1,
                            rand: Random = new Random,
                            tau: Double,
                            rho: Double = 0.7): Tree[T] = {
    val size = data.size
    if (size == 0) {
      Empty[T]
    } else if (size <= leafSize) {
      Leaf(data)
    } else {
      val randomPivot = data(rand.nextInt(size)).vectorWithNorm
      val leftPivot = data.maxBy(v => randomPivot.fastSquaredDistance(v.vectorWithNorm)).vectorWithNorm
      if (leftPivot == randomPivot) {
        // all points are identical (or only one point left)
        Leaf(data, randomPivot, 0.0)
      } else {
        val rightPivot = data.maxBy(v => leftPivot.fastSquaredDistance(v.vectorWithNorm)).vectorWithNorm
        val pivot = new VectorWithNorm(Vectors.fromBreeze((leftPivot.vector.toBreeze + rightPivot.vector.toBreeze) / 2.0))
        val radius = math.sqrt(data.map(v => pivot.fastSquaredDistance(v.vectorWithNorm)).max)
        val dataWithDistance = data.map(v =>
          (v, leftPivot.fastDistance(v.vectorWithNorm), rightPivot.fastDistance(v.vectorWithNorm))
        )
        // implemented boundary is parabola (rather than perpendicular plane described in the paper)
        val leftPartition = dataWithDistance.filter { case (_, left, right) => left - right <= 2 * tau }.map(_._1)
        val rightPartition = dataWithDistance.filter { case (_, left, right) => right - left <= 2 * tau }.map(_._1)

        if(leftPartition.size > size * rho || rightPartition.size > size * rho) {
          //revert back to metric node
          val (leftPartition, rightPartition) = data.partition{
            v => leftPivot.fastSquaredDistance(v.vectorWithNorm) < rightPivot.fastSquaredDistance(v.vectorWithNorm)
          }
          MetricTree(
            build(leftPartition, leafSize, rand, tau, rho),
            leftPivot,
            build(rightPartition, leafSize, rand, tau, rho),
            rightPivot,
            pivot,
            radius
          )
        } else {
          SpillTree(
            build(leftPartition, leafSize, rand, tau, rho),
            leftPivot,
            build(rightPartition, leafSize, rand, tau, rho),
            rightPivot,
            pivot,
            radius,
            tau,
            leftPartition.size + rightPartition.size - size
          )
        }
      }
    }
  }
}

/**
 * Structure to maintain search progress/results for a single query vector.
 * Internally uses a PriorityQueue to maintain a max-heap to keep track of the
 * next neighbor to evict.
 *
 * @param queryVector vector being searched
 * @param k number of neighbors to return
 */
private[knn]
class KNNCandidates[T <: hasVector](val queryVector: VectorWithNorm, val k: Int) extends Serializable {
  private[knn] val candidates = mutable.PriorityQueue.empty[(T, Double)] {
    Ordering.by(_._2)
  }

  // return the current maximum distance from neighbor to search vector
  def maxDistance: Double = if(candidates.isEmpty) 0.0 else candidates.head._2
  // insert evict neighbor if required. however it doesn't make sure the insert improves
  // search results. it is caller's responsibility to make sure either candidate list
  // is not full or the inserted neighbor brings the maxDistance down
  def insert(v: T, d: Double): Unit = {
    while(candidates.size >= k) candidates.dequeue()
    candidates.enqueue((v, d))
  }
  def insert(v: T): Unit = insert(v, v.vectorWithNorm.fastDistance(queryVector))
  def tryInsert(v: T): Unit = {
    val distance = v.vectorWithNorm.fastDistance(queryVector)
    if(notFull || distance < maxDistance) insert(v, distance)
  }
  def toIterable: Iterable[T] = candidates.map(_._1)
  def notFull: Boolean = candidates.size < k
}
