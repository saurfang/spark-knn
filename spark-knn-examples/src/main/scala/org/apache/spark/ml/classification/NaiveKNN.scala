package org.apache.spark.ml.classification

import org.apache.spark.SparkException
import org.apache.spark.ml.knn.KNN.{RowWithVector, VectorWithNorm}
import org.apache.spark.ml.knn.{KNNModel, KNNParams}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.ml.{Model, Predictor}
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{ArrayType, DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._

import scala.collection.mutable.ArrayBuffer

/**
  * Brute-force kNN with k = 1
  */
class NaiveKNNClassifier(override val uid: String) extends Predictor[Vector, NaiveKNNClassifier, NaiveKNNClassifierModel] {
  def this() = this(Identifiable.randomUID("naiveknnc"))

  override def copy(extra: ParamMap): NaiveKNNClassifier = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): NaiveKNNClassifierModel = {
    // Extract columns from data.  If dataset is persisted, do not persist oldDataset.
    val instances = extractLabeledPoints(dataset).map {
      case LabeledPoint(label: Double, features: Vector) => (label, features)
    }
    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    val labelSummarizer = instances.treeAggregate(new MultiClassSummarizer)(
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

    val points = instances.map{
      case (label, features) => (label, new VectorWithNorm(features))
    }

    new NaiveKNNClassifierModel(uid, points, numClasses)
  }

}

class NaiveKNNClassifierModel(
                               override val uid: String,
                               val points: RDD[(Double, VectorWithNorm)],
                               val _numClasses: Int) extends ProbabilisticClassificationModel[Vector, NaiveKNNClassifierModel] {
  override def numClasses: Int = _numClasses

  override def transform(dataset: Dataset[_]): DataFrame = {
    import dataset.sparkSession.implicits._

    val features = dataset.select($(featuresCol))
      .map(r => new VectorWithNorm(r.getAs[Vector](0)))

    val merged = features.rdd.zipWithUniqueId()
      .cartesian(points)
      .map {
        case ((u, i), (label, v)) =>
            val dist = u.fastSquaredDistance(v)
            (i, (dist, label))
      }
      .topByKey(1)(Ordering.by(e => -e._1))
      .map{
        case (id, labels) =>
          val vector = new Array[Double](numClasses)
          var i = 0
          while (i < labels.length) {
            vector(labels(i)._2.toInt) += 1
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

          (id, values.toSeq)
      }

    dataset.sqlContext.createDataFrame(
      dataset.toDF().rdd.zipWithUniqueId().map { case (row, i) => (i, row) }
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

  override def copy(extra: ParamMap): NaiveKNNClassifierModel = {
    val copied = new NaiveKNNClassifierModel(uid, points, numClasses)
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

  override def predictRaw(features: Vector): Vector = {
    throw new SparkException("predictRaw function should not be called directly since kNN prediction is done in distributed fashion. Use transform instead.")
  }
}
