package org.apache.spark.mllib.knn

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.{Partition, TaskContext}

class KNNRDD(val rootTree: Tree,
             @transient childrenTree: RDD[Tree]) extends RDD[Vector](childrenTree) {

  @DeveloperApi
  override def compute(split: Partition, context: TaskContext): Iterator[Vector] =
    firstParent[Tree].iterator(split, context).flatMap(_.iterator)

  override protected def getPartitions: Array[Partition] = firstParent.partitions

  override def getPreferredLocations(split: Partition): Seq[String] =
    firstParent.preferredLocations(split)
}
