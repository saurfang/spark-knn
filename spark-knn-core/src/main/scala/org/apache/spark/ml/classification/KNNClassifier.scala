package org.apache.spark.ml.classification

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.knn._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkException

import scala.collection.mutable.ArrayBuffer

/**
  * [[https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm]] for classification.
  * An object is classified by a majority vote of its neighbors, with the object being assigned to
  * the class most common among its k nearest neighbors.
  */
class KNNClassifier(override val uid: String) extends ProbabilisticClassifier[Vector, KNNClassifier, KNNClassificationModel]
with KNNParams with HasWeightCol {

  def this() = this(Identifiable.randomUID("knnc"))

  /** @group setParam */
  override def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  override def setLabelCol(value: String): this.type = {
    set(labelCol, value)

    if ($(weightCol).isEmpty) {
      set(inputCols, Array(value))
    } else {
      set(inputCols, Array(value, $(weightCol)))
    }
  }

  //fill in default label col
  setDefault(inputCols, Array($(labelCol)))

  /** @group setWeight */
  def setWeightCol(value: String): this.type = {
    set(weightCol, value)

    if (value.isEmpty) {
      set(inputCols, Array($(labelCol)))
    } else {
      set(inputCols, Array($(labelCol), value))
    }
  }

  setDefault(weightCol -> "")

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

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

  override protected def train(dataset: Dataset[_]): KNNClassificationModel = {
    // Extract columns from data.  If dataset is persisted, do not persist oldDataset.
    val instances = extractLabeledPoints(dataset).map {
      case LabeledPoint(label: Double, features: Vector) => (label, features)
    }
    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    val labelSummarizer = instances.treeAggregate(
      new MultiClassSummarizer)(
      seqOp = (c, v) => (c, v) match {
        case (labelSummarizer: MultiClassSummarizer, (label: Double, features: Vector)) =>
          labelSummarizer.add(label)
      },
      combOp = (c1, c2) => (c1, c2) match {
        case (classSummarizer1: MultiClassSummarizer, classSummarizer2: MultiClassSummarizer) =>
          classSummarizer1.merge(classSummarizer2)
      })

    val histogram = labelSummarizer.histogram
    val numInvalid = labelSummarizer.countInvalid
    val numClasses = histogram.length

    if (numInvalid != 0) {
      val msg = s"Classification labels should be in {0 to ${numClasses - 1} " +
        s"Found $numInvalid invalid labels."
      logError(msg)
      throw new SparkException(msg)
    }

    val knnModel = copyValues(new KNN()).fit(dataset)
    knnModel.toNewClassificationModel(uid, numClasses)
  }

  override def fit(dataset: Dataset[_]): KNNClassificationModel = {
    // Need to overwrite this method because we need to manually overwrite the buffer size
    // because it is not supposed to stay the same as the Classifier if user sets it to -1.
    transformSchema(dataset.schema, logging = true)
    val model = train(dataset)
    val bufferSize = model.getBufferSize
    copyValues(model.setParent(this)).setBufferSize(bufferSize)
  }

  override def copy(extra: ParamMap): KNNClassifier = defaultCopy(extra)
}

class KNNClassificationModel private[ml](
                                          override val uid: String,
                                          val topTree: Broadcast[Tree],
                                          val subTrees: RDD[Tree],
                                          val _numClasses: Int
                                        ) extends ProbabilisticClassificationModel[Vector, KNNClassificationModel]
with KNNModelParams with HasWeightCol with Serializable {
  require(subTrees.getStorageLevel != StorageLevel.NONE,
    "KNNModel is not designed to work with Trees that have not been cached")

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setBufferSize(value: Double): this.type = set(bufferSize, value)

  override def numClasses: Int = _numClasses

  //TODO: This can benefit from DataSet API
  override def transform(dataset: Dataset[_]): DataFrame = {
    val getWeight: Row => Double = {
      if($(weightCol).isEmpty) {
        r => 1.0
      } else {
        r => r.getDouble(1)
      }
    }

    val neighborRDD : RDD[(Long, Array[(Row, Double)])] = transform(dataset, topTree, subTrees)
    val merged = neighborRDD
      .map {
        case (id, labelsDists) =>
          val (labels, _) = labelsDists.unzip
          val vector = new Array[Double](numClasses)
          var i = 0
          while (i < labels.length) {
            vector(labels(i).getDouble(0).toInt) += getWeight(labels(i))
            i += 1
          }
          val rawPrediction = Vectors.dense(vector)
          lazy val probability = raw2probability(rawPrediction)
          lazy val prediction = probability2prediction(probability)

          val values = new ArrayBuffer[Any]
          if ($(rawPredictionCol).nonEmpty) {
            values.append(rawPrediction)
          }
          if ($(probabilityCol).nonEmpty) {
            values.append(probability)
          }
          if ($(predictionCol).nonEmpty) {
            values.append(prediction)
          }

          (id, values)
      }

    dataset.sqlContext.createDataFrame(
      dataset.toDF().rdd.zipWithIndex().map { case (row, i) => (i, row) }
        .leftOuterJoin(merged) //make sure we don't lose any observations
        .map {
        case (i, (row, values)) => Row.fromSeq(row.toSeq ++ values.get)
      },
      transformSchema(dataset.schema)
    )
  }

  override def transformSchema(schema: StructType): StructType = {
    var transformed = schema
    if ($(rawPredictionCol).nonEmpty) {
      transformed = SchemaUtils.appendColumn(transformed, $(rawPredictionCol), new VectorUDT)
    }
    if ($(probabilityCol).nonEmpty) {
      transformed = SchemaUtils.appendColumn(transformed, $(probabilityCol), new VectorUDT)
    }
    if ($(predictionCol).nonEmpty) {
      transformed = SchemaUtils.appendColumn(transformed, $(predictionCol), DoubleType)
    }
    transformed
  }

  override def copy(extra: ParamMap): KNNClassificationModel = {
    val copied = new KNNClassificationModel(uid, topTree, subTrees, numClasses)
    copyValues(copied, extra).setParent(parent)
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        var i = 0
        val size = dv.size

        var sum = 0.0
        while (i < size) {
          sum += dv.values(i)
          i += 1
        }

        i = 0
        while (i < size) {
          dv.values(i) /= sum
          i += 1
        }

        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in KNNClassificationModel:" +
          " raw2probabilitiesInPlace encountered SparseVector")
    }
  }

  override protected def predictRaw(features: Vector): Vector = {
    throw new SparkException("predictRaw function should not be called directly since kNN prediction is done in distributed fashion. Use transform instead.")
  }
}
