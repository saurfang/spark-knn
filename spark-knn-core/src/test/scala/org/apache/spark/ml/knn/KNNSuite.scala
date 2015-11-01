package org.apache.spark.ml.knn

import org.apache.spark.SharedSparkContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.knn.KNN.VectorWithNorm
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SQLContext}
import org.scalatest.{FunSuite, Matchers}

import scala.collection.mutable

class KNNSuite extends FunSuite with SharedSparkContext with Matchers {

  private[this] val data = (-10 to 10).flatMap(i => (-10 to 10).map(j => Vectors.dense(i, j)))
  private[this] val leafSize = 5

  test("KNN can be fitted") {
    val knn = new KNN()
      .setTopTreeSize(data.size / 10)
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setAuxCols(Array("features"))

    val sqlContext = new SQLContext(sc)
    val rdd = sc.parallelize(data.map(v => Row(v.toArray: _*)))
    val assembler = new VectorAssembler()
      .setInputCols(Array("x", "y"))
      .setOutputCol("features")
    val df = assembler.transform(
      sqlContext.createDataFrame(rdd,
        StructType(
          Seq(
            StructField("x", DoubleType),
            StructField("y", DoubleType)
          )
        )
      )
    )
    val model = knn.fit(df).setK(1)

    val results = model.transform(df)
    results.count shouldBe data.size

    results.collect().foreach {
      row =>
        val vector = row.getAs[Vector](2)
        val neighbors = row.getAs[mutable.WrappedArray[Row]](3)
        if(neighbors.isEmpty) {
          println(vector)
        }
        neighbors.length shouldBe 1
        val neighbor = neighbors.head.getAs[Vector](0)
        new VectorWithNorm(vector).fastSquaredDistance(new VectorWithNorm(neighbor)) should be <= 2.0
    }
  }
}
