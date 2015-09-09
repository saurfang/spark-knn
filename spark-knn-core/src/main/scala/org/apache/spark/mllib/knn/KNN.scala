package org.apache.spark.mllib.knn

import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD


class KNN private(

                   ) extends Serializable with Logging {
  def run(data: RDD[Vector]): KNNRDD = ???
}
