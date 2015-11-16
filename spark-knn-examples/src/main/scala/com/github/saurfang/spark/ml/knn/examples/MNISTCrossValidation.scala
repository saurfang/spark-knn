package com.github.saurfang.spark.ml.knn.examples

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.KNNClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{Logging, SparkConf, SparkContext}

object MNISTCrossValidation extends Logging {
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

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
    logInfo(s"In-sample: $insample, Out-of-sample: $outofsample")
    logInfo(s"Cross-validated: ${cvModel.avgMetrics.toSeq}")
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
