package org.apache.spark.ml.knn

import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.classification.KNNClassifier
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.knn.KNN.VectorWithNorm
import org.apache.spark.ml.regression.KNNRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.log4j

import scala.collection.mutable
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers


class KNNSuite extends AnyFunSuite with Matchers  {

  val logger = log4j.Logger.getLogger(getClass)

  val spark = SparkSession.builder()
    .master("local")
    .getOrCreate()

  val sc = spark.sparkContext

  private[this] val data = (-10 to 10).flatMap(i => (-10 to 10).map(j => Vectors.dense(i, j)))
  private[this] val leafSize = 5

  test("KNN can be fitted") {
    val distance = EuclideanDistanceMetric
    val knn = new KNN()
      .setTopTreeSize(data.size / 10)
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setAuxCols(Array("features"))

    val df = createDataFrame()
    val model = knn.fit(df).setK(1)

    val results = model.transform(df).collect()
    results.length shouldBe data.size

    results.foreach {
      row =>
        val vector = row.getAs[Vector](3)
        val neighbors = row.getAs[mutable.WrappedArray[Row]](4)
        if (neighbors.isEmpty) {
          logger.error(vector.toString)
        }
        neighbors.length shouldBe 1
        val neighbor = neighbors.head.getAs[Vector](0)
        distance.fastSquaredDistance(new VectorWithNorm(vector), new VectorWithNorm(neighbor)) shouldBe 0.0
    }
  }

  test("KNN fits correctly with maxDistance") {
    val distanceMetric = EuclideanDistanceMetric
    val knn = new KNN()
      .setTopTreeSize(data.size / 10)
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setAuxCols(Array("features"))

    val df = createDataFrame()
    val model = knn.fit(df).setK(6).setMaxDistance(1)

    val results = model.transform(df).collect()
    results.length shouldBe data.size

    results.foreach {
      row =>
        val vector = row.getAs[Vector](3)
        val neighbors = row.getAs[mutable.WrappedArray[Row]](4)
        if (neighbors.isEmpty) {
          logger.error(vector.toString)
        }

        val numEdges = vector.toArray.map(math.abs).count(_ == 10)
        if (neighbors.length > 5 - numEdges) {
          logger.error(vector.toString)
          logger.error(neighbors.toList.toString)
        }
        neighbors.length should be <= 5 - numEdges

        val closest = neighbors.head.getAs[Vector](0)
        distanceMetric.fastSquaredDistance(new VectorWithNorm(vector), new VectorWithNorm(closest)) shouldBe 0.0
        val rest = neighbors.tail.map(_.getAs[Vector](0))
        rest.foreach { neighbor =>
          val sqDist = distanceMetric.fastSquaredDistance(new VectorWithNorm(vector), new VectorWithNorm(neighbor))
          sqDist shouldEqual 1.0 +- 1e-6
        }
    }
  }

  test("KNN returns correct distance column values") {
    val distanceMetric = EuclideanDistanceMetric
    val knn = new KNN()
      .setTopTreeSize(data.size / 10)
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setAuxCols(Array("features"))

    val df = createDataFrame()
    val model = knn.fit(df).setK(6).setMaxDistance(1).setNeighborsCol("neighbors").setDistanceCol("distance")

    val results = model.transform(df).collect()
    results.length shouldBe data.size

    results.foreach {
      row =>
        val vector = row.getAs[Vector](row.fieldIndex("features"))
        val neighbors = row.getAs[mutable.WrappedArray[Row]](row.fieldIndex("neighbors"))
        val distances = row.getAs[mutable.WrappedArray[Double]](row.fieldIndex("distance"))
        if (neighbors.isEmpty) {
          logger.error(vector.toString)
        }

        val numEdges = vector.toArray.map(math.abs).count(_ == 10)
        if (neighbors.length > 5 - numEdges) {
          logger.error(vector.toString)
          logger.error(neighbors.toList.toString)
        }
        neighbors.length should be <= 5 - numEdges

        if (distances.length != neighbors.length) {
          logger.error(vector.toString)
          logger.error(neighbors.toList.toString)
          logger.error(distances.toList.toString)
        }
        distances.length should be (neighbors.length)

        val closest = neighbors.head.getAs[Vector](0)
        val closestDist = distances.head
        val closestCalDist = distanceMetric.fastSquaredDistance(new VectorWithNorm(vector), new VectorWithNorm(closest))
        closestCalDist shouldEqual closestDist

        val rest = neighbors.tail.map(_.getAs[Vector](0)).zip(distances.tail.toList)
        rest.foreach {
          case (neighbor, distance) =>
            val sqDist = distanceMetric.fastSquaredDistance(new VectorWithNorm(vector), new VectorWithNorm(neighbor))
            sqDist shouldEqual 1.0 +- 1e-6
            sqDist shouldEqual distance +- 1e-6
        }
    }
  }

  test("KNNClassifier can be fitted with/without weight column") {
    val knn = new KNNClassifier()
      .setTopTreeSize(data.size / 10)
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(1)
    checkKNN(knn.fit)
    checkKNN(knn.setWeightCol("z").fit)
  }

  test("KNNRegressor can be fitted with/without weight column") {
    val knn = new KNNRegression()
      .setTopTreeSize(data.size / 10)
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(1)
    checkKNN(knn.fit)
    checkKNN(knn.setWeightCol("z").fit)
  }

  test("KNNParams are copied correctly") {
    val knn = new KNNClassifier()
      .setTopTreeSize(data.size / 10)
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(2)
    val model = knn.fit(createDataFrame().withColumn("label", lit(1.0)))
    // check pre-set parameters are correctly copied
    model.getK shouldBe 2
    // check auto generated buffer size is correctly transferred
    model.getBufferSize should be > 0.0
  }

  test("Errors with helpful message if size of data is smaller than topTreeSize") {
    val knn = new KNNClassifier()
      .setTopTreeSize(data.size + 1)
      .setK(2)

    val thrown = intercept[Exception] {
      knn.fit(createDataFrame().withColumn("label", lit(1.0)))
    }

    assert(thrown.getMessage == "Invalid top tree size relative to size of data. " +
      "Data to fit of size 441 was less than topTreeSize 442")
  }

  test("BufferSize is not estimated if rho = 0") {
    val knn = new KNNClassifier()
      .setTopTreeSize(data.size / 10)
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setBalanceThreshold(0)
    val model = knn.fit(createDataFrame().withColumn("label", lit(1.0)))
    model.getBufferSize shouldBe 0.0
  }

  private[this] def checkKNN(fit: DataFrame => PredictionModel[_, _]): Unit = {
    val df = createDataFrame()
    df.sqlContext.udf.register("label", { v: Vector => math.abs(v(0)) })
    val training = df.selectExpr("*", "label(features) as label")
    val model = fit(training)

    val results = model.transform(training).select("label", "prediction").collect()
    results.length shouldBe data.size

    results foreach {
      row => row.getDouble(0) shouldBe row.getDouble(1)
    }
  }

  private[this] def createDataFrame(): DataFrame = {
    val rdd = sc.parallelize(data.map(v => Row(v.toArray: _*)))
    val assembler = new VectorAssembler()
      .setInputCols(Array("x", "y"))
      .setOutputCol("features")
    assembler.transform(
      spark.createDataFrame(rdd,
        StructType(
          Seq(
            StructField("x", DoubleType),
            StructField("y", DoubleType)
          )
        )
      ).withColumn("z", lit(1.0))
    )
  }
}
