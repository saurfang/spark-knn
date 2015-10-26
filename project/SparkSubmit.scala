import sbtsparksubmit.SparkSubmitPlugin.autoImport._

object SparkSubmit {
  lazy val settings =
    SparkSubmitSetting("sparkMNIST",
      Seq(
        "--master", "local[3]",
        "--class", "com.github.saurfang.spark.mllib.knn.examples.MNIST"
      )
    )
}