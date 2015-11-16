package com.github.saurfang.spark.ml.knn.examples

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.classification.{NaiveKNNClassifier, KNNClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.{IntParam, ParamMap}
import org.apache.spark.ml.tuning.{Benchmarker, ParamGridBuilder}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{Logging, SparkConf, SparkContext}

/**
  * Benchmark KNN as a function of number of observations
  */
object MNISTBenchmark extends Logging {
  def main(args: Array[String]) {
    val ns = if(args.isEmpty) (2500 to 10000 by 2500).toArray else args.map(_.toInt)

    val conf = new SparkConf()
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    //read in raw label and features
    val dataset = MLUtils.loadLibSVMFile(sc, "data/mnist/mnist.bz2")
      .zipWithIndex()
      .sortBy(_._2, numPartitions = 10)
      .keys
      .toDF()
      .cache()
    dataset.count() //force persist

    val limiter = new Limiter()
    val knn = new KNNClassifier()
      .setTopTreeSize(dataset.count().toInt / 1000)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setK(1)
    val naiveKNN = new NaiveKNNClassifier()

    val pipeline = new Pipeline()
      .setStages(Array(limiter, knn))
    val naivePipeline = new Pipeline()
      .setStages(Array(limiter, naiveKNN))

    val paramGrid = new ParamGridBuilder()
      .addGrid(limiter.n, ns)
      .build()

    val bm = new Benchmarker()
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumTimes(3)

    val bmModel = bm.setEstimator(pipeline).fit(dataset)
    val naiveBMModel = bm.setEstimator(naivePipeline).fit(dataset)
    logInfo(s"knn: ${bmModel.avgTrainingRuntimes.toSeq} / ${bmModel.avgEvaluationRuntimes.toSeq}")
    logInfo(s"naive: ${naiveBMModel.avgTrainingRuntimes.toSeq} / ${naiveBMModel.avgEvaluationRuntimes.toSeq}")
  }
}

class Limiter(override val uid: String) extends Transformer {
  def this() = this(Identifiable.randomUID("limiter"))

  val n: IntParam = new IntParam(this, "n", "number of rows to limit")

  def setN(value: Int): this.type = set(n, value)

  override def transform(dataset: DataFrame): DataFrame = dataset.limit($(n))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = schema
}
