package com.github.saurfang.spark.ml.knn.examples

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.KNNClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.log4j

object MNISTCrossValidation {

  val logger = log4j.Logger.getLogger(getClass)

  def main(args: Array[String]) {
    val spark = SparkSession.builder().getOrCreate()
    val sc = spark.sparkContext
    import spark.implicits._

    //read in raw label and features
    val dataset = MLUtils.loadLibSVMFile(sc, "data/mnist/mnist.bz2")
      .toDF()
      //.limit(10000)

    //split traning and testing
    val Array(train, test) = dataset.randomSplit(Array(0.7, 0.3), seed = 1234L).map(_.cache())

    //create PCA matrix to reduce feature dimensions
    val pca = new PCA()
      .setInputCol("features")
      .setK(50)
      .setOutputCol("pcaFeatures")
    val knn = new KNNClassifier()
      .setTopTreeSize(50)
      .setFeaturesCol("pcaFeatures")
      .setPredictionCol("prediction")
      .setK(1)

    val pipeline = new Pipeline()
      .setStages(Array(pca, knn))

    val paramGrid = new ParamGridBuilder()
//      .addGrid(knn.k, 1 to 20)
      .addGrid(pca.k, 10 to 100 by 10)
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val cvModel = cv.fit(train)

    val insample = validate(cvModel.transform(train))
    val outofsample = validate(cvModel.transform(test))

    //reference accuracy: in-sample 95% out-of-sample 94%
    logger.info(s"In-sample: $insample, Out-of-sample: $outofsample")
    logger.info(s"Cross-validated: ${cvModel.avgMetrics.toSeq}")
  }

  private[this] def validate(results: DataFrame): Double = {
    results
      .selectExpr("SUM(CASE WHEN label = prediction THEN 1.0 ELSE 0.0 END) / COUNT(1)")
      .collect()
      .head
      .getDecimal(0)
      .doubleValue()
  }

}
