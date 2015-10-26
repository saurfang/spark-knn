package com.github.saurfang.spark.mllib.knn.examples

import org.apache.spark.mllib.knn.{KNN, hasVector}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{Logging, SparkConf, SparkContext}

object MNIST extends Logging {
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)

    val dataset = sc.textFile("data/MNIST/mnist.csv.gz")
      .zipWithIndex()
      .filter(_._2 < 10000)
      .sortBy(_._2, true, 10)
      .map(_._1)
      .map(_.split(","))
      .map(x => new LabeledVector(Vectors.dense(x.tail.map(_.toDouble)), x.head.toInt))
      .cache()

    val Array(train, test) = dataset.randomSplit(Array(0.7, 0.3)).map(_.cache())

    val knn = new KNN(500, 10, 10).run(train)

    val k = 5
    val insample = validate(knn.query(train, k))
    val outofsample = validate(knn.query(test, k))

    logInfo(s"In-sample: $insample, Out-of-sample: $outofsample")
  }

  private[this] def validate(results: RDD[(LabeledVector, Iterable[LabeledVector])]): Double = {
    results.filter {
      case (p, candidates) =>
        val predicated = candidates.map(_.label).groupBy(k => k).map{ case (l, itr) => (l, itr.size)}.maxBy(_._2)._1
        p.label == predicated
    }.count().toDouble / results.count()
  }
}

class LabeledVector(val v: Vector, val label: Int) extends hasVector {
  override def vector: Vector = v
}
