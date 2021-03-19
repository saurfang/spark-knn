package org.apache.spark.ml.knn

import breeze.linalg.{DenseVector, Vector => BV}
import breeze.stats._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.classification.KNNClassificationModel
import org.apache.spark.ml.knn.KNN.{KNNPartitioner, RowWithVector, VectorWithNorm}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.regression.KNNRegressionModel
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._
import org.apache.spark.rdd.{RDD, ShuffledRDD}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.random.XORShiftRandom
import org.apache.spark.{HashPartitioner, Partitioner}
import org.apache.log4j
import org.apache.spark.mllib.knn.KNNUtils

import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer
import scala.util.hashing.byteswap64

// features column => vector, input columns => auxiliary columns to return by KNN model
private[ml] trait KNNModelParams extends Params with HasFeaturesCol with HasInputCols {
  /**
    * Param for the column name for returned neighbors.
    * Default: "neighbors"
    *
    * @group param
    */
  val neighborsCol = new Param[String](this, "neighborsCol", "column names for returned neighbors")

  /** @group getParam */
  def getNeighborsCol: String = $(neighborsCol)

  /**
    * Param for distance column that will create a distance column of each nearest neighbor
    * Default: no distance column will be used
    *
    * @group param
    */
  val distanceCol = new Param[String](this, "distanceCol", "column that includes each neighbors' distance as an additional column")

  /** @group getParam */
  def getDistanceCol: String = $(distanceCol)

  /**
    * Param for number of neighbors to find (> 0).
    * Default: 5
    *
    * @group param
    */
  val k = new IntParam(this, "k", "number of neighbors to find", ParamValidators.gt(0))

  /** @group getParam */
  def getK: Int = $(k)

  /**
   * Param for maximum distance to find neighbors
   * Default: Double.PositiveInfinity
    *
    * @group param
   */
  val maxDistance = new DoubleParam(this, "maxNeighbors", "maximum distance to find neighbors", // todo: maxDistance or maxNeighbors?
                                     ParamValidators.gt(0))

  /** @group getParam */
  def getMaxDistance: Double = $(maxDistance)

  /**
    * Param for size of buffer used to construct spill trees and top-level tree search.
    * Note the buffer size is 2 * tau as described in the paper.
    *
    * When buffer size is 0.0, the tree itself reverts to a metric tree.
    * -1.0 triggers automatic effective nearest neighbor distance estimation.
    *
    * Default: -1.0
    *
    * @group param
    */
  val bufferSize = new DoubleParam(this, "bufferSize",
    "size of buffer used to construct spill trees and top-level tree search", ParamValidators.gtEq(-1.0))

  /** @group getParam */
  def getBufferSize: Double = $(bufferSize)

  private[ml] def transform(data: RDD[Vector], topTree: Broadcast[Tree], subTrees: RDD[Tree]): RDD[(Long, Array[(Row,Double)])] = {
    val searchData = data.zipWithIndex()
      .flatMap {
        case (vector, index) =>
          val vectorWithNorm = new VectorWithNorm(vector)
          val idx = KNN.searchIndices(vectorWithNorm, topTree.value, $(bufferSize))
            .map(i => (i, (vectorWithNorm, index)))

          assert(idx.nonEmpty, s"indices must be non-empty: $vector ($index)")
          idx
      }
      .partitionBy(new HashPartitioner(subTrees.partitions.length))

    // for each partition, search points within corresponding child tree
    val results = searchData.zipPartitions(subTrees) {
      (childData, trees) =>
        val tree = trees.next()
        assert(!trees.hasNext)
        childData.flatMap {
          case (_, (point, i)) =>
            tree.query(point, $(k)).collect {
              case (neighbor, distance) if distance <= $(maxDistance) =>
                (i, (neighbor.row, distance))
            }
        }
    }

    // merge results by point index together and keep topK results
    results.topByKey($(k))(Ordering.by(-_._2))
      .map { case (i, seq) => (i, seq) }
  }

  private[ml] def transform(dataset: Dataset[_], topTree: Broadcast[Tree], subTrees: RDD[Tree]): RDD[(Long, Array[(Row, Double)])] = {
    transform(dataset.select($(featuresCol)).rdd.map(_.getAs[Vector](0)), topTree, subTrees)
  }

}

private[ml] trait KNNParams extends KNNModelParams with HasSeed {
  /**
    * Param for number of points to sample for top-level tree (> 0).
    * Default: 1000
    *
    * @group param
    */
  val topTreeSize = new IntParam(this, "topTreeSize", "number of points to sample for top-level tree", ParamValidators.gt(0))

  /** @group getParam */
  def getTopTreeSize: Int = $(topTreeSize)

  /**
    * Param for number of points at which to switch to brute-force for top-level tree (> 0).
    * Default: 5
    *
    * @group param
    */
  val topTreeLeafSize = new IntParam(this, "topTreeLeafSize",
    "number of points at which to switch to brute-force for top-level tree", ParamValidators.gt(0))

  /** @group getParam */
  def getTopTreeLeafSize: Int = $(topTreeLeafSize)

  /**
    * Param for number of points at which to switch to brute-force for distributed sub-trees (> 0).
    * Default: 20
    *
    * @group param
    */
  val subTreeLeafSize = new IntParam(this, "subTreeLeafSize",
    "number of points at which to switch to brute-force for distributed sub-trees", ParamValidators.gt(0))

  /** @group getParam */
  def getSubTreeLeafSize: Int = $(subTreeLeafSize)

  /**
    * Param for number of sample sizes to take when estimating buffer size (at least two samples).
    * Default: 100 to 1000 by 100
    *
    * @group param
    */
  val bufferSizeSampleSizes = new IntArrayParam(this, "bufferSizeSampleSize",  // todo: should this have an 's' at the end?
    "number of sample sizes to take when estimating buffer size", { arr: Array[Int] => arr.length > 1 && arr.forall(_ > 0) })

  /** @group getParam */
  def getBufferSizeSampleSizes: Array[Int] = $(bufferSizeSampleSizes)

  /**
    * Param for fraction of total points at which spill tree reverts back to metric tree
    * if either child contains more points (0 <= rho <= 1).
    * Default: 70%
    *
    * @group param
    */
  val balanceThreshold = new DoubleParam(this, "balanceThreshold",
    "fraction of total points at which spill tree reverts back to metric tree if either child contains more points",
    ParamValidators.inRange(0, 1))

  /** @group getParam */
  def getBalanceThreshold: Double = $(balanceThreshold)

  setDefault(topTreeSize -> 1000, topTreeLeafSize -> 10, subTreeLeafSize -> 30,
    bufferSize -> -1.0, bufferSizeSampleSizes -> (100 to 1000 by 100).toArray, balanceThreshold -> 0.7,
    k -> 5, neighborsCol -> "neighbors", distanceCol -> "", maxDistance -> Double.PositiveInfinity)

  /**
    * Validates and transforms the input schema.
    *
    * @param schema input schema
    * @return output schema
    */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    val auxFeatures = $(inputCols).map(c => schema(c))
    val schemaWithNeighbors = SchemaUtils.appendColumn(schema, $(neighborsCol), ArrayType(StructType(auxFeatures)))

    if ($(distanceCol).isEmpty) {
      schemaWithNeighbors
    } else {
      SchemaUtils.appendColumn(schemaWithNeighbors, $(distanceCol), ArrayType(DoubleType))
    }
  }
}

/**
  * kNN Model facilitates k-Nestrest Neighbor search by storing distributed hybrid spill tree.
  * Top level tree is a MetricTree but instead of using back tracking, it searches all possible leaves in parallel
  * to avoid multiple iterations. It uses the same buffer size that is used in model training, when the search
  * vector falls into the buffer zone of the node, it dispatches search to both children.
  *
  * A high level overview of the search phases is as follows:
  *
  *  1. For each vector to search, go through the top level tree to output a pair of (index, point)
  *  1. Repartition search points by partition index
  *  1. Search each point through the hybrid spill tree in that particular partition
  *  1. For each point, merge results from different partitions and keep top k results.
  *
  */
class KNNModel private[ml](
                            override val uid: String,
                            val topTree: Broadcast[Tree],
                            val subTrees: RDD[Tree]
                          ) extends Model[KNNModel] with KNNModelParams {
  require(subTrees.getStorageLevel != StorageLevel.NONE,
    "KNNModel is not designed to work with Trees that have not been cached")

  /** @group setParam */
  def setNeighborsCol(value: String): this.type = set(neighborsCol, value)

  /** @group setParam */
  def setDistanceCol(value: String): this.type = set(distanceCol, value)

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setMaxDistance(value: Double): this.type = set(maxDistance, value)

  /** @group setParam */
  def setBufferSize(value: Double): this.type = set(bufferSize, value)

  //TODO: All these can benefit from DataSet API
  override def transform(dataset: Dataset[_]): DataFrame = {
    val merged: RDD[(Long, Array[(Row,Double)])] = transform(dataset, topTree, subTrees)

    val withDistance = $(distanceCol).nonEmpty

    dataset.sqlContext.createDataFrame(
      dataset.toDF().rdd.zipWithIndex().map { case (row, i) => (i, row) }
        .leftOuterJoin(merged)
        .map {
          case (i, (row, neighborsAndDistances)) =>
            val (neighbors, distances) = neighborsAndDistances.map(_.unzip).getOrElse((Array.empty[Row], Array.empty[Double]))
            if (withDistance) {
              Row.fromSeq(row.toSeq :+ neighbors :+ distances)
            } else {
              Row.fromSeq(row.toSeq :+ neighbors)
            }
        },
      transformSchema(dataset.schema)
    )
  }

  override def transformSchema(schema: StructType): StructType = {
    val auxFeatures = $(inputCols).map(c => schema(c))
    val schemaWithNeighbors = SchemaUtils.appendColumn(schema, $(neighborsCol), ArrayType(StructType(auxFeatures)))
    if ($(distanceCol).isEmpty) {
      schemaWithNeighbors
    } else {
      SchemaUtils.appendColumn(schemaWithNeighbors, $(distanceCol), ArrayType(DoubleType))
    }
  }

  override def copy(extra: ParamMap): KNNModel = {
    val copied = new KNNModel(uid, topTree, subTrees)
    copyValues(copied, extra).setParent(parent)
  }

  def toNewClassificationModel(uid: String, numClasses: Int): KNNClassificationModel = {
    copyValues(new KNNClassificationModel(uid, topTree, subTrees, numClasses))
  }

  def toNewRegressionModel(uid: String): KNNRegressionModel = {
    copyValues(new KNNRegressionModel(uid, topTree, subTrees))
  }
}

/**
  * k-Nearest Neighbors (kNN) algorithm
  *
  * kNN finds k closest observations in training dataset. It can be used for both classification and regression.
  * Furthermore it can also be used for other purposes such as input to clustering algorithm.
  *
  * While the brute-force approach requires no pre-training, each prediction requires going through the entire training
  * set resulting O(n log(k)) runtime per individual prediction using a heap keep track of neighbor candidates.
  * Many different implementations have been proposed such as Locality Sensitive Hashing (LSH), KD-Tree, Metric Tree and etc.
  * Each algorithm has its shortcomings that prevent them to be effective on large-scale and/or high-dimensional dataset.
  *
  * This is an implementation of kNN based upon distributed Hybrid Spill-Trees where training points are organized into
  * distributed binary trees. The algorithm is designed to support accurate approximate kNN search but by tuning parameters
  * an exact search can also be performed with cost of additional runtime.
  *
  * Each binary tree node is either a
  *
  * '''Metric Node''':
  * Metric Node partition points exclusively into two children by finding two pivot points and divide by middle plane.
  * When searched, the child whose pivot is closer to query vector is searched first. Back tracking is required to
  * ensure accuracy in this case, where the other child should be searched if it can possibly contain better neighbor
  * based upon candidates picked during previous search.
  *
  * '''Spill Node''':
  * Spill Node also partitions points into two children however there are an overlapping buffer between the two pivot
  * points. The larger the buffer size, the less effective the node eliminates points thus could increase tree height.
  * When searched, defeatist search is used where only one child is searched and no back tracking happens in this
  * process. Because of the buffer between two children, we are likely to end up with good enough candidates without
  * searching the other part of the tree.
  *
  * While Spill Node promises O(h) runtime where h is the tree height, the tree is deeper than Metric Tree's O(log n)
  * height on average. Furthermore, when it comes down to leaves where points are more closer to each other, the static
  * buffer size means more points will end up in the buffer. Therefore a Balance Threshold (rho) is introduced: when
  * either child of Spill Node makes up more than rho fraction of the total points at this level, Spill Node is reverted
  * back to a Metric Node.
  *
  * A high level overview of the algorithm is as follows:
  *
  *  1. Sample M data points (M is relatively small and can be held in driver)
  *  1. Build the top level metric tree
  *  1. Repartition RDD by assigning each point to leaf node of the above tree
  *  1. Build a hybrid spill tree at each partition
  *
  * This concludes the training phase of kNN.
  * See [[KNNModel]] for details on prediction phase.
  *
  *
  * This algorithm is described in [[http://dx.doi.org/10.1109/WACV.2007.18]] where it was shown to scale well in terms of
  * number of observations and dimensions, bounded by the available memory across clusters (billions in paper's example).
  * This implementation adapts the MapReduce algorithm to work with Spark.
  *
  */
class KNN(override val uid: String) extends Estimator[KNNModel] with KNNParams {
  def this() = this(Identifiable.randomUID("knn"))

  /** @group setParam */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setAuxCols(value: Array[String]): this.type = set(inputCols, value)

  /** @group setParam */
  def setTopTreeSize(value: Int): this.type = set(topTreeSize, value)

  /** @group setParam */
  def setTopTreeLeafSize(value: Int): this.type = set(topTreeLeafSize, value)

  /** @group setParam */
  def setSubTreeLeafSize(value: Int): this.type = set(subTreeLeafSize, value)

  /** @group setParam */
  def setBufferSizeSampleSizes(value: Array[Int]): this.type = set(bufferSizeSampleSizes, value)

  /** @group setParam */
  def setBalanceThreshold(value: Double): this.type = set(balanceThreshold, value)

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  override def fit(dataset: Dataset[_]): KNNModel = {
    val rand = new XORShiftRandom($(seed))
    //prepare data for model estimation
    val data = dataset.selectExpr($(featuresCol), $(inputCols).mkString("struct(", ",", ")"))
      .rdd
      .map(row => new RowWithVector(row.getAs[Vector](0), row.getStruct(1)))
    //sample data to build top-level tree
    val sampled = data.sample(withReplacement = false, $(topTreeSize).toDouble / dataset.count(), rand.nextLong()).collect()
    val topTree = MetricTree.build(sampled, $(topTreeLeafSize), rand.nextLong())
    //build partitioner using top-level tree
    val part = new KNNPartitioner(topTree)
    //noinspection ScalaStyle
    val repartitioned = new ShuffledRDD[RowWithVector, Null, Null](data.map(v => (v, null)), part).keys

    val tau =
      if ($(balanceThreshold) > 0 && $(bufferSize) < 0) {
        KNN.estimateTau(data, $(bufferSizeSampleSizes), rand.nextLong())
      } else {
        math.max(0, $(bufferSize))
      }
    logInfo("Tau is: " + tau)

    val trees = repartitioned.mapPartitionsWithIndex {
      (partitionId, itr) =>
        val rand = new XORShiftRandom(byteswap64($(seed) ^ partitionId))
        val childTree =
          HybridTree.build(itr.toIndexedSeq, $(subTreeLeafSize), tau, $(balanceThreshold), rand.nextLong())

        Iterator(childTree)
    }.persist(StorageLevel.MEMORY_AND_DISK)
    // TODO: force persisting trees primarily for benchmark. any reason not to do this for regular runs?
    trees.count()

    val model = new KNNModel(uid, trees.context.broadcast(topTree), trees).setParent(this)
    copyValues(model).setBufferSize(tau)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): KNN = defaultCopy(extra)
}


object KNN {

  val logger = log4j.Logger.getLogger(classOf[KNN])

  /**
    * VectorWithNorm can use more efficient algorithm to calculate distance
    */
  case class VectorWithNorm(vector: Vector, norm: Double) {
    def this(vector: Vector) = this(vector, Vectors.norm(vector, 2))

    def this(vector: BV[Double]) = this(Vectors.fromBreeze(vector))

    def fastSquaredDistance(v: VectorWithNorm): Double = {
      KNNUtils.fastSquaredDistance(vector, norm, v.vector, v.norm)
    }

    def fastDistance(v: VectorWithNorm): Double = math.sqrt(fastSquaredDistance(v))
  }

  /**
    * VectorWithNorm plus auxiliary row information
    */
  case class RowWithVector(vector: VectorWithNorm, row: Row) {
    def this(vector: Vector, row: Row) = this(new VectorWithNorm(vector), row)
  }

  /**
    * Estimate a suitable buffer size based on dataset
    *
    * A suitable buffer size is the minimum size such that nearest neighbors can be accurately found even at
    * boundary of splitting plane between pivot points. Therefore assuming points are uniformly distributed in
    * high dimensional space, it should be approximately the average distance between points.
    *
    * Specifically the number of points within a certain radius of a given point is proportionally to the density of
    * points raised to the effective number of dimensions, of which manifold data points exist on:
    * R_s = \frac{c}{N_s ** 1/d}
    * where R_s is the radius, N_s is the number of points, d is effective number of dimension, and c is a constant.
    *
    * To estimate R_s_all for entire dataset, we can take samples of the dataset of different size N_s to compute R_s.
    * We can estimate c and d using linear regression. Lastly we can calculate R_s_all using total number of observation
    * in dataset.
    *
    */
  def estimateTau(data: RDD[RowWithVector], sampleSize: Array[Int], seed: Long): Double = {
    val total = data.count()

    // take samples of points for estimation
    val samples = data.mapPartitionsWithIndex {
      case (partitionId, itr) =>
        val rand = new XORShiftRandom(byteswap64(seed ^ partitionId))
        itr.flatMap {
          p => sampleSize.zipWithIndex
            .filter { case (size, _) => rand.nextDouble() * total < size }
            .map { case (size, index) => (index, p) }
        }
    }
    // compute N_s and R_s pairs
    val estimators = samples
      .groupByKey()
      .map {
        case (index, points) => (points.size, computeAverageDistance(points))
      }.collect().distinct

    // collect x and y vectors
    val x = DenseVector(estimators.map { case (n, _) => math.log(n) })
    val y = DenseVector(estimators.map { case (_, d) => math.log(d) })

    // estimate log(R_s) = alpha + beta * log(N_s)
    val xMeanVariance = meanAndVariance(x)
    val xmean = xMeanVariance.mean
    val yMeanVariance = meanAndVariance(y)
    val ymean = yMeanVariance.mean

    val corr = (mean(x *:* y) - xmean * ymean) / math.sqrt((mean(x *:* x) - xmean * xmean) * (mean(y *:* y) - ymean * ymean))

    val beta = corr * yMeanVariance.stdDev / xMeanVariance.stdDev
    val alpha = ymean - beta * xmean
    val rs = math.exp(alpha + beta * math.log(total))

    if (beta > 0 || beta.isNaN || rs.isNaN) {
      val yMax = breeze.linalg.max(y)
      logger.error(
        s"""Unable to estimate Tau with positive beta: $beta. This maybe because data is too small.
            |Setting to $yMax which is the maximum average distance we found in the sample.
            |This may leads to poor accuracy. Consider manually set bufferSize instead.
            |You can also try setting balanceThreshold to zero so only metric trees are built.""".stripMargin)
      yMax
    } else {
      // c = alpha, d = - 1 / beta
      rs / math.sqrt(-1 / beta)
    }
  }

  // compute the average distance of nearest neighbors within points using brute-force
  private[this] def computeAverageDistance(points: Iterable[RowWithVector]): Double = {
    val distances = points.map {
      point => points.map(p => p.vector.fastSquaredDistance(point.vector)).filter(_ > 0).min
    }.map(math.sqrt)

    distances.sum / distances.size
  }

  /**
    * Search leaf index used by KNNPartitioner to partition training points
    *
    * @param v one training point to partition
    * @param tree top tree constructed using sampled points
    * @param acc accumulator used to help determining leaf index
    * @return leaf/partition index
    */
  @tailrec
  private[knn] def searchIndex(v: RowWithVector, tree: Tree, acc: Int = 0): Int = {
    tree match {
      case node: MetricTree =>
        val leftDistance = node.leftPivot.fastSquaredDistance(v.vector)
        val rightDistance = node.rightPivot.fastSquaredDistance(v.vector)
        if (leftDistance < rightDistance) {
          searchIndex(v, node.leftChild, acc)
        } else {
          searchIndex(v, node.rightChild, acc + node.leftChild.leafCount)
        }
      case _ => acc // reached leaf
    }
  }

  //TODO: Might want to make this tail recursive
  private[ml] def searchIndices(v: VectorWithNorm, tree: Tree, tau: Double, acc: Int = 0): Seq[Int] = {
    tree match {
      case node: MetricTree =>
        val leftDistance = node.leftPivot.fastDistance(v)
        val rightDistance = node.rightPivot.fastDistance(v)

        val buffer = new ArrayBuffer[Int]
        if (leftDistance - rightDistance <= tau) {
          buffer ++= searchIndices(v, node.leftChild, tau, acc)
        }

        if (rightDistance - leftDistance <= tau) {
          buffer ++= searchIndices(v, node.rightChild, tau, acc + node.leftChild.leafCount)
        }

        buffer
      case _ => Seq(acc) // reached leaf
    }
  }

  /**
    * Partitioner used to map vector to leaf node which determines the partition it goes to
    *
    * @param tree `Tree` used to find leaf
    */
  class KNNPartitioner[T <: RowWithVector](tree: Tree) extends Partitioner {
    override def numPartitions: Int = tree.leafCount

    override def getPartition(key: Any): Int = {
      key match {
        case v: RowWithVector => searchIndex(v, tree)
        case _ => throw new IllegalArgumentException(s"Key must be of type Vector but got: $key")
      }
    }

  }

}
