package com.github.saurfang.spark.mllib.knn.examples

import org.apache.spark.ml.knn.KNN
import org.apache.spark.mllib.KNNUDT.myVectorUDT
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{Logging, SparkConf, SparkContext}

object MNIST extends Logging {
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val dataset =
      sqlContext.createDataFrame(
        sc.textFile("data/MNIST/mnist.csv.gz")
          .zipWithIndex()
          .filter(_._2 < 10000)
          .sortBy(_._2, true, 10)
          .map(_._1)
          .map(_.split(","))
          .map(x => Row(Vectors.dense(x.tail.map(_.toDouble)), x.head.toInt)),
        StructType(
          Seq(
            StructField("features", new myVectorUDT),
            StructField("label", IntegerType)
          )
        )
      )

    val Array(train, test) = dataset.randomSplit(Array(0.7, 0.3)).map(_.cache())

    val knn = new KNN().setTopTreeSize(100).setAuxCols(Array("label")).fit(train).setK(10)

    sqlContext.udf.register("predict", {
      neighbors: Seq[Row] =>
        if (neighbors.isEmpty) {
          None
        } else {
          Some(neighbors.map(_.getInt(0)).groupBy(k => k).map { case (l, itr) => (l, itr.size) }.maxBy(_._2)._1)
        }
    })
    val insample = validate(knn.transform(train))
    val outofsample = validate(knn.transform(test))

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
