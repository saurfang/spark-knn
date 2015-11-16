import sbtsparksubmit.SparkSubmitPlugin.autoImport._

object SparkSubmit {
  lazy val settings =
    SparkSubmitSetting(
      SparkSubmitSetting("sparkMNIST",
        Seq(
          "--master", "local[3]",
          "--class", "com.github.saurfang.spark.ml.knn.examples.MNIST"
        )
      ),
      SparkSubmitSetting("sparkMNISTCross",
        Seq(
          "--master", "local[3]",
          "--class", "com.github.saurfang.spark.ml.knn.examples.MNISTCrossValidation"
        )
      ),
      SparkSubmitSetting("sparkMNISTBench",
        Seq(
          "--master", "local[3]",
          "--class", "com.github.saurfang.spark.ml.knn.examples.MNISTBenchmark"
        )
      )
    )
}
