package org.apache.spark.ml.regression

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.knn.{KNN, KNNModelParams, KNNParams, Tree}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel

/**
  * [[https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm]] for regression.
  * The output value is simply the average of the values of its k nearest neighbors.
  */
class KNNRegression(override val uid: String) extends Predictor[Vector, KNNRegression, KNNRegressionModel]
with KNNParams with HasWeightCol {
  def this() = this(Identifiable.randomUID("knnr"))

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

  override protected def train(dataset: Dataset[_]): KNNRegressionModel = {
    val knnModel = copyValues(new KNN()).fit(dataset)
    knnModel.toNewRegressionModel(uid)
  }

  override def fit(dataset: Dataset[_]): KNNRegressionModel = {
    // Need to overwrite this method because we need to manually overwrite the buffer size
    // because it is not supposed to stay the same as the Regressor if user sets it to -1.
    transformSchema(dataset.schema, logging = true)
    val model = train(dataset)
    val bufferSize = model.getBufferSize
    copyValues(model.setParent(this)).setBufferSize(bufferSize)
  }

  override def copy(extra: ParamMap): KNNRegression = defaultCopy(extra)
}

class KNNRegressionModel private[ml](
                                      override val uid: String,
                                      val topTree: Broadcast[Tree],
                                      val subTrees: RDD[Tree]
                                    ) extends PredictionModel[Vector, KNNRegressionModel]
with KNNModelParams with HasWeightCol with Serializable {
  require(subTrees.getStorageLevel != StorageLevel.NONE,
    "KNNModel is not designed to work with Trees that have not been cached")

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setBufferSize(value: Double): this.type = set(bufferSize, value)

  //TODO: This can benefit from DataSet API in Spark 1.6
  override def transformImpl(dataset: Dataset[_]): DataFrame = {
    val getWeight: Row => Double = {
      if($(weightCol).isEmpty) {
        r => 1.0
      } else {
        r => r.getDouble(1)
      }
    }

    val neighborDataset : RDD[(Long, Array[(Row, Double)])] = transform(dataset, topTree, subTrees)
    val merged = neighborDataset
      .map {
        case (id, labelsDists) =>
          val (labels, _) = labelsDists.unzip
          var i = 0
          var weight = 0.0
          var sum = 0.0
          val length = labels.length
          while (i < length) {
            val row = labels(i)
            val w = getWeight(row)
            sum += row.getDouble(0) * w
            weight += w
            i += 1
          }

          (id, sum / weight)
      }

    dataset.sqlContext.createDataFrame(
      dataset.toDF().rdd.zipWithIndex().map { case (row, i) => (i, row) }
        .leftOuterJoin(merged) //make sure we don't lose any observations
        .map {
        case (i, (row, value)) => Row.fromSeq(row.toSeq :+ value.get)
      },
      transformSchema(dataset.schema)
    )
  }

  override def copy(extra: ParamMap): KNNRegressionModel = {
    val copied = new KNNRegressionModel(uid, topTree, subTrees)
    copyValues(copied, extra).setParent(parent)
  }

  override def predict(features: Vector): Double = {
    val neighborDataset : RDD[(Long, Array[(Row, Double)])] = transform(subTrees.context.parallelize(Seq(features)), topTree, subTrees)
    val results = neighborDataset.first()._2
    val labels = results.map(_._1.getDouble(0))
    labels.sum / labels.length
  }
}
