package org.apache.spark.ml.knn

import org.apache.spark.ml.classification.KNNClassifier
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.knn.KNN.VectorWithNorm
import org.apache.spark.ml.regression.KNNRegression
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{Logging, SharedSparkContext}
import org.scalatest.{FunSuite, Matchers}

import scala.collection.mutable

class KNNSuite extends FunSuite with SharedSparkContext with Matchers with Logging {

  private[this] val data = (-10 to 10).flatMap(i => (-10 to 10).map(j => Vectors.dense(i, j)))
  private[this] val leafSize = 5

  test("KNN can be fitted") {
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
        val vector = row.getAs[Vector](2)
        val neighbors = row.getAs[mutable.WrappedArray[Row]](3)
        if (neighbors.isEmpty) {
          logError(vector.toString)
        }
        neighbors.length shouldBe 1
        val neighbor = neighbors.head.getAs[Vector](0)
        new VectorWithNorm(vector).fastSquaredDistance(new VectorWithNorm(neighbor)) shouldBe 0.0
    }
  }

  test("KNNClassifier can be fitted") {
    val knn = new KNNClassifier()
      .setTopTreeSize(data.size / 10)
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)

    val df = createDataFrame()
    df.sqlContext.udf.register("label", { v: Vector => math.abs(v(0)) })
    val training = df.selectExpr("*", "label(features) as label")
    val model = knn.fit(training).setK(1)

    val results = model.transform(training).select("label", "prediction").collect()
    results.length shouldBe data.size

    results foreach {
      row => row.getDouble(0) shouldBe row.getDouble(1)
    }
  }

  test("KNNRegressor can be fitted") {
    val knn = new KNNRegression()
      .setTopTreeSize(data.size / 10)
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)

    val df = createDataFrame()
    df.sqlContext.udf.register("label", { v: Vector => math.abs(v(0)) })
    val training = df.selectExpr("*", "label(features) as label")
    val model = knn.fit(training).setK(1)

    val results = model.transform(training).select("label", "prediction").collect()
    results.length shouldBe data.size

    results foreach {
      row => row.getDouble(0) shouldBe row.getDouble(1)
    }
  }

  private[this] def createDataFrame(): DataFrame = {
    val sqlContext = new SQLContext(sc)
    val rdd = sc.parallelize(data.map(v => Row(v.toArray: _*)))
    val assembler = new VectorAssembler()
      .setInputCols(Array("x", "y"))
      .setOutputCol("features")
    assembler.transform(
      sqlContext.createDataFrame(rdd,
        StructType(
          Seq(
            StructField("x", DoubleType),
            StructField("y", DoubleType)
          )
        )
      )
    )
  }
}
