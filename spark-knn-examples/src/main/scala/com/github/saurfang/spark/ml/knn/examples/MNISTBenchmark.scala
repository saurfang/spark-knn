package com.github.saurfang.spark.ml.knn.examples

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.classification.{KNNClassifier, NaiveKNNClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.{IntParam, ParamMap}
import org.apache.spark.ml.tuning.{Benchmarker, ParamGridBuilder}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j

import scala.collection.mutable

/**
  * Benchmark KNN as a function of number of observations
  */
object MNISTBenchmark {

  val logger = log4j.Logger.getLogger(getClass)

  def main(args: Array[String]) {
    val ns = if(args.isEmpty) (2500 to 10000 by 2500).toArray else args(0).split(',').map(_.toInt)
    val path = if(args.length >= 2) args(1) else "data/mnist/mnist.bz2"
    val numPartitions = if(args.length >= 3) args(2).toInt else 10
    val models = if(args.length >=4) args(3).split(',') else Array("tree","naive")

    val conf = new SparkConf()
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    //read in raw label and features
    val dataset = MLUtils.loadLibSVMFile(sc, path)
      .zipWithIndex()
      .filter(_._2 < ns.max)
      .sortBy(_._2, numPartitions = numPartitions)
      .keys
      .toDF()
      .cache()
    dataset.count() //force persist

    val limiter = new Limiter()
    val knn = new KNNClassifier()
      .setTopTreeSize(numPartitions * 10)
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

    val metrics = mutable.ArrayBuffer[String]()
    if(models.contains("tree")) {
      val bmModel = bm.setEstimator(pipeline).fit(dataset)
      metrics += s"knn: ${bmModel.avgTrainingRuntimes.toSeq} / ${bmModel.avgEvaluationRuntimes.toSeq}"
    }
    if(models.contains("naive")) {
      val naiveBMModel = bm.setEstimator(naivePipeline).fit(dataset)
      metrics += s"naive: ${naiveBMModel.avgTrainingRuntimes.toSeq} / ${naiveBMModel.avgEvaluationRuntimes.toSeq}"
    }
    logger.info(metrics.mkString("\n"))
  }
}

class Limiter(override val uid: String) extends Transformer {
  def this() = this(Identifiable.randomUID("limiter"))

  val n: IntParam = new IntParam(this, "n", "number of rows to limit")

  def setN(value: Int): this.type = set(n, value)

  // hack to maintain number of partitions (otherwise it collapses to 1 which is unfair for naiveKNN)
  override def transform(dataset: Dataset[_]): DataFrame = dataset.limit($(n)).repartition(dataset.rdd.partitions.length).toDF()

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = schema
}
