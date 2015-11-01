package com.github.saurfang.spark.mllib.knn.examples

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{PCA, VectorAssembler}
import org.apache.spark.ml.knn.KNN
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{Logging, SparkConf, SparkContext}

object MNIST extends Logging {
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    //read in raw label and features
    val rdd =
      sc.textFile("data/MNIST/mnist.csv.gz")
        .zipWithIndex()
        .filter(_._2 < 10000)
        .sortBy(_._2, true, 10)
        .map(_._1)
        .map(_.split(","))
        .map(x => Row(x.map(_.toDouble): _*))
    val featureCols = (1 until rdd.first().length).map("col" + _)

    //assemble features into vector
    val assembler = new VectorAssembler()
      .setInputCols(featureCols.toArray)
      .setOutputCol("features")
    val dataset =
      assembler.transform(
        sqlContext.createDataFrame(rdd,
          StructType(
            StructField("label", DoubleType) +:
              featureCols.map(x => StructField(x, DoubleType))
          )
        )
      )
        .select("label", "features")

    //split traning and testing
    val Array(train, test) = dataset.randomSplit(Array(0.7, 0.3)).map(_.cache())

    //create PCA matrix to reduce feature dimensions
    val pca = new PCA()
      .setInputCol("features")
      .setK(100)
      .setOutputCol("pcaFeatures")
    val knn = new KNN()
      .setTopTreeSize(100)
      .setFeaturesCol("pcaFeatures")
      .setAuxCols(Array("label"))
      .setK(10)
    val pipeline = new Pipeline()
      .setStages(Array(pca, knn))
      .fit(train)

    //register udf that predicts based on neighbors' labels
    sqlContext.udf.register("predict", {
      neighbors: Seq[Row] =>
        if (neighbors.isEmpty) {
          None
        } else {
          Some(neighbors.map(_.getDouble(0)).groupBy(k => k).map { case (l, itr) => (l, itr.size) }.maxBy(_._2)._1)
        }
    })
    val insample = validate(pipeline.transform(train))
    val outofsample = validate(pipeline.transform(test))

    //reference accuracy: in-sample 95% out-of-sample 94%
    logInfo(s"In-sample: $insample, Out-of-sample: $outofsample")
  }

  private[this] def validate(results: DataFrame): Double = {
    results.selectExpr("*", "predict(neighbors) as predicted")
      .selectExpr("SUM(CASE WHEN label = predicted THEN 1.0 ELSE 0.0 END) / COUNT(1)")
      .collect()
      .head
      .getDecimal(0)
      .doubleValue()
  }
}
