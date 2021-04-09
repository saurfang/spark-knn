package org.apache.spark.ml.regression

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class KNNRegressionSuite extends AnyFunSuite with Matchers {

  val spark = SparkSession.builder()
    .master("local")
    .getOrCreate()

  import spark.implicits._
  val rawDF1 = Seq(
    (9.5,37.48596719,-122.2428196, 1.0),
    (9.5,37.49115273,-122.2319523, 2.0),
    (9.5,37.49099652,-122.2324551, 3.0),
    (9.5,37.4886712,-122.2348786, 1.0),
    (9.5,37.48696518,-122.2384678, 3.0),
    (9.5,37.48473396,-122.2345444, 3.0),
    (9.5,37.48565758,-122.2412995, 2.0),
    (9.5,37.48033504,-122.2364642, 2.0)
  ).toDF("col1", "col2", "col3", "label")
  val rawDF2 = Seq(
    (9.5,37.48495049,-122.2335112)
  ).toDF("col1", "col2", "col3")

  val assembler = new VectorAssembler()
    .setInputCols(Array("col1", "col2", "col3"))
    .setOutputCol("features")
  val trainDF = assembler.transform(rawDF1)
  val testDF = assembler.transform(rawDF2)

  test("KNNRegression can be fitted using euclidean distance") {
    val knnr = new KNNRegression()
      .setTopTreeSize(5)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setLabelCol("label")
      .setSeed(31)
      .setK(5)
    val knnModel = knnr.fit(trainDF)
    val outputDF = knnModel.transform(testDF)
    val predictions = outputDF.collect().map(_.getAs[Double]("prediction"))
    predictions shouldEqual Array(2.4)
  }

  test("KNNRegression can be fitted using nan_euclidean distance") {
    val knnr = new KNNRegression()
      .setTopTreeSize(5)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setLabelCol("label")
      .setSeed(31)
      .setK(5)
    knnr.set(knnr.metric, "nan_euclidean")
    val knnModel = knnr.fit(trainDF)
    val outputDF = knnModel.transform(testDF)
    val predictions = outputDF.collect().map(_.getAs[Double]("prediction"))
    predictions shouldEqual Array(2.4)
  }
}
