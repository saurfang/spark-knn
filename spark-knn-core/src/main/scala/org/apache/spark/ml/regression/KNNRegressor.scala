package org.apache.spark.ml.regression

import org.apache.spark.Logging
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.knn.{KNN, KNNParams, Tree}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.storage.StorageLevel

/**
  * [[https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm]] for regression.
  * The output value is simply the average of the values of its k nearest neighbors.
  */
class KNNRegression(override val uid: String) extends Predictor[Vector, KNNRegression, KNNRegressionModel]
with KNNParams with Logging {
  def this() = this(Identifiable.randomUID("knnr"))

  /** @group setParam */
  override def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  override def setLabelCol(value: String): this.type = {
    set(labelCol, value)
    set(inputCols, Array(value))
  }

  //fill in default label col
  setDefault(inputCols, Array($(labelCol)))

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

  override protected def train(dataset: DataFrame): KNNRegressionModel = {
    val knnModel = copyValues(new KNN()).fit(dataset)
    val model = new KNNRegressionModel(uid, knnModel.topTree, knnModel.subTrees)
    copyValues(model).setBufferSize(knnModel.getBufferSize)
  }

  override def copy(extra: ParamMap): KNNRegression = defaultCopy(extra)
}

class KNNRegressionModel private[ml](
                                      override val uid: String,
                                      val topTree: Broadcast[Tree],
                                      val subTrees: RDD[Tree]
                                    ) extends PredictionModel[Vector, KNNRegressionModel]
with KNNParams with Serializable {
  require(subTrees.getStorageLevel != StorageLevel.NONE,
    "KNNModel is not designed to work with Trees that have not been cached")

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setBufferSize(value: Double): this.type = set(bufferSize, value)

  //TODO: This can benefit from DataSet API in Spark 1.6
  override def transformImpl(dataset: DataFrame): DataFrame = {
    val merged = transform(dataset, topTree, subTrees)
      .map {
        case (id, labels) =>
          var i = 0
          var sum = 0.0
          val length = labels.length
          while (i < length) {
            sum += labels(i).getDouble(0)
            i += 1
          }

          (id, sum / length)
      }

    dataset.sqlContext.createDataFrame(
      dataset.rdd.zipWithIndex().map { case (row, i) => (i, row) }
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

  override protected def predict(features: Vector): Double = {
    val results = transform(subTrees.context.parallelize(Seq(features)), topTree, subTrees).first()._2
    val labels = results.map(_.getDouble(0))
    labels.sum / labels.length
  }
}
