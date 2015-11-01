package org.apache.spark.ml.knn

import breeze.linalg.{DenseVector, Vector => BV}
import breeze.stats._
import org.apache.spark.ml.knn.KNN.{VectorWithNorm, KNNPartitioner, RowWithVector}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.knn.KNNUtils
import org.apache.spark.mllib.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._
import org.apache.spark.rdd.{RDD, ShuffledRDD}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.random.XORShiftRandom
import org.apache.spark.{HashPartitioner, Logging, Partitioner}

import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer
import scala.util.hashing.byteswap64

// features column => vector, input columns => auxiliary columns to return by KNN model
private[knn] trait KNNModelParams extends Params with HasFeaturesCol with HasInputCols {
  /**
   * Param for the column name for returned neighbors.
   * Default: "neighbors"
   * @group param
   */
  val neighborsCol = new Param[String](this, "neighborsCol", "column names for returned neighbors")

  /** @group getParam */
  def getNeighborsCol: String = $(neighborsCol)

  /**
   * Param for number of neighbors to find (> 0).
   * Default: 5
   * @group param
   */
  val k = new IntParam(this, "k", "number of neighbors to find", ParamValidators.gt(0))

  /** @group getParam */
  def getK: Int = $(k)

  /**
   * Param for size of buffer used to construct spill trees and top-level tree search.
   * Note the buffer size is 2 * tau as described in the paper.
   * When buffer size is 0.0, the tree itself reverts to a metric tree.
   * -1.0 triggers automatic effective nearest neighbor distance estimation.
   * Default: -1.0
   * @group param
   */
  val bufferSize = new DoubleParam(this, "bufferSize",
    "size of buffer used to construct spill trees and top-level tree search", ParamValidators.gtEq(-1.0))

  /** @group getParam */
  def getBufferSize: Double = $(bufferSize)
}

private[knn] trait KNNParams extends KNNModelParams with HasSeed {
  /**
   * Param for number of points to sample for top-level tree (> 0).
   * Default: 1000
   * @group param
   */
  val topTreeSize = new IntParam(this, "topTreeSize", "number of points to sample for top-level tree", ParamValidators.gt(0))

  /** @group getParam */
  def getTopTreeSize: Int = $(topTreeSize)

  /**
   * Param for number of points at which to switch to brute-force for top-level tree (> 0).
   * Default: 5
   * @group param
   */
  val topTreeLeafSize = new IntParam(this, "topTreeLeafSize",
    "number of points at which to switch to brute-force for top-level tree", ParamValidators.gt(0))

  /** @group getParam */
  def getTopTreeLeafSize: Int = $(topTreeLeafSize)

  /**
   * Param for number of points at which to switch to brute-force for distributed sub-trees (> 0).
   * Default: 20
   * @group param
   */
  val subTreeLeafSize = new IntParam(this, "subTreeLeafSize",
    "number of points at which to switch to brute-force for distributed sub-trees", ParamValidators.gt(0))

  /** @group getParam */
  def getSubTreeLeafSize: Int = $(subTreeLeafSize)

  /**
   * Param for number of sample sizes to take when estimating buffer size (at least two samples).
   * Default: 100 to 1000 by 100
   * @group param
   */
  val bufferSizeSampleSizes = new Param[Array[Int]](this, "bufferSizeSampleSize",
    "number of sample sizes to take when estimating buffer size",
    {arr: Array[Int] => arr.length > 1 && arr.forall(_ > 0)})

  /** @group getParam */
  def getBufferSizeSampleSizes: Array[Int] = $(bufferSizeSampleSizes)

  /**
   * Param for fraction of total points at which spill tree reverts back to metric tree
   * if either child contains more points (0 <= rho <= 1).
   * Default: 70%
   * @group param
   */
  val balanceThreshold = new DoubleParam(this, "balanceThreshold",
    "fraction of total points at which spill tree reverts back to metric tree if either child contains more points",
    ParamValidators.inRange(0, 1))

  /** @group getParam */
  def getBalanceThreshold: Double = $(balanceThreshold)

  setDefault(topTreeSize -> 1000, topTreeLeafSize -> 5, subTreeLeafSize -> 20,
    bufferSize -> -1.0, bufferSizeSampleSizes -> (100 to 1000 by 100).toArray, balanceThreshold -> 0.7,
    k -> 5, neighborsCol -> "neighbors")

  /**
   * Validates and transforms the input schema.
   * @param schema input schema
   * @return output schema
   */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    val auxFeatures = $(inputCols).map(c => schema(c))
    SchemaUtils.appendColumn(schema, $(neighborsCol), ArrayType(StructType(auxFeatures)))
  }
}

class KNNModel private[ml] (
                             override val uid: String,
                           val topTree: Tree,
                           val subTrees: RDD[Tree]
                             ) extends Model[KNNModel] with KNNModelParams {
  require(subTrees.getStorageLevel != StorageLevel.NONE,
    "KNNModel is not designed to work with Trees that have not been cached")

  /** @group setParam */
  def setNeighborsCol(value: String): this.type = set(neighborsCol, value)

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setBufferSize(value: Double): this.type = set(bufferSize, value)

  //TODO: All these can benefit from DataSet API in Spark 1.6
  override def transform(dataset: DataFrame): DataFrame = {
    val searchData = dataset.select($(featuresCol)).rdd.map(_.getAs[Vector](0)).zipWithIndex()
      .flatMap{
        point =>
          val idx = KNN.searchIndecies(new VectorWithNorm(point._1), topTree, $(bufferSize)).map(i => (i, point))
          assert(idx.nonEmpty, s"indices must be non-empty: $point")
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
            val vectorWithNorm = new VectorWithNorm(point)
            tree.query(vectorWithNorm, $(k)).map {
              neighbor => (i, (neighbor.row, neighbor.vector.fastSquaredDistance(vectorWithNorm)))
            }
        }
    }

    // merge results by point index together and keep topK results
    val merged = results.topByKey($(k))(Ordering.by(- _._2))
      .map { case (i, seq) => (i, seq.map(_._1)) }

    dataset.sqlContext.createDataFrame(
      dataset.rdd.zipWithIndex().map{ case (row, i) => (i, row) }
        .leftOuterJoin(merged)
        .map {
          case (i, (row, neighbors)) =>
            Row.fromSeq(row.toSeq :+ neighbors.getOrElse(Array.empty))
        },
      transformSchema(dataset.schema)
    )
  }

  override def transformSchema(schema: StructType): StructType = {
    val auxFeatures = $(inputCols).map(c => schema(c))
    SchemaUtils.appendColumn(schema, $(neighborsCol), ArrayType(StructType(auxFeatures)))
  }

  override def copy(extra: ParamMap): KNNModel = {
    val copied = new KNNModel(uid, topTree, subTrees)
    copyValues(copied, extra).setParent(parent)
  }
}

class KNN (override val uid: String) extends Estimator[KNNModel] with KNNParams {
  def this() = this(Identifiable.randomUID("knn"))

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

  override def fit(dataset: DataFrame): KNNModel = {
    val rand = new XORShiftRandom($(seed))
    val data = dataset.selectExpr($(featuresCol), $(inputCols).mkString("struct(", ",", ")"))
      .map(row => new RowWithVector(row.getAs[Vector](0), row.getStruct(1)))
    val sampled = data.sample(false, $(topTreeSize).toDouble / dataset.count(), rand.nextLong()).collect()
    val topTree = MetricTree.build(sampled, $(topTreeLeafSize), rand.nextLong())
    val part = new KNNPartitioner(topTree)
    val repartitioned = new ShuffledRDD[RowWithVector, Null, Null](data.map(v => (v, null)), part)

    val tau =
      if($(bufferSize) < 0) {
        KNN.estimateTau(data, $(bufferSizeSampleSizes), rand.nextLong())
      } else {
        $(bufferSize)
      }
    logInfo("Tau is: " + tau)

    val trees = repartitioned.mapPartitionsWithIndex{
      (partitionId, itr) =>
        val rand = new XORShiftRandom(byteswap64($(seed) ^ partitionId))
        val childTree =
          HybridTree.build(itr.map(_._1).toIndexedSeq, $(subTreeLeafSize), tau, $(balanceThreshold), rand.nextLong())

        Iterator(childTree)
    }.persist(StorageLevel.MEMORY_AND_DISK)

    val model = new KNNModel(uid, topTree, trees).setParent(this)
    copyValues(model).setBufferSize(tau)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): KNN = defaultCopy(extra)
}


object KNN extends Logging {
  case class VectorWithNorm(vector: Vector, norm: Double) {
    def this(vector: Vector) = this(vector, Vectors.norm(vector, 2))
    def this(vector: BV[Double]) = this(Vectors.fromBreeze(vector))

    def fastSquaredDistance(v: VectorWithNorm): Double = {
      KNNUtils.fastSquaredDistance(vector, norm, v.vector, v.norm)
    }
    def fastDistance(v: VectorWithNorm): Double = math.sqrt(fastSquaredDistance(v))
  }

  case class RowWithVector(vector: VectorWithNorm, row: Row) {
    def this(vector: Vector, row: Row) = this(new VectorWithNorm(vector), row)
  }

  def estimateTau(data: RDD[RowWithVector], sampleSize: Array[Int], seed: Long): Double = {
    val total = data.count()

    val estimators = data.mapPartitionsWithIndex {
      case (partitionId, itr) =>
        val rand = new XORShiftRandom(byteswap64(seed ^ partitionId))
        itr.flatMap{
          p => sampleSize.zipWithIndex
            .filter{ case (size, _) => rand.nextDouble() * total < size }
            .map{ case (size, index) => (index, p) }
        }
    }
      .groupByKey()
      .map {
        case (index, points) => (points.size, computeAverageDistance(points))
      }.collect().distinct

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

    rs / math.sqrt(- 1 / beta)
  }

  private[this] def computeAverageDistance(points: Iterable[RowWithVector]): Double = {
    val distances = points.map{
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
        if(leftDistance < rightDistance) {
          searchIndex(v, node.leftChild, acc)
        } else {
          searchIndex(v, node.rightChild, acc + node.leftChild.leafCount)
        }
      case _ => acc // reached leaf
    }
  }

  //TODO: Might want to make this tail recursive
  private[knn] def searchIndecies(v: VectorWithNorm, tree: Tree, tau: Double, acc: Int = 0): Seq[Int] = {
    tree match {
      case node: MetricTree =>
        val leftDistance = node.leftPivot.fastDistance(v)
        val rightDistance = node.rightPivot.fastDistance(v)

        val buffer = new ArrayBuffer[Int]
        if(leftDistance - rightDistance <= tau) {
          buffer ++= searchIndecies(v, node.leftChild, tau, acc)
        }

        if (rightDistance - leftDistance <= tau) {
          buffer ++= searchIndecies(v, node.rightChild, tau, acc + node.leftChild.leafCount)
        }

        buffer
      case _ => Seq(acc) // reached leaf
    }
  }

  /**
   * Partitioner used to map vector to leaf node which determines the partition it goes to
   *
   * @param tree [[MetricTree]] used to find leaf
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
