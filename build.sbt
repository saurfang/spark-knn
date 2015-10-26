import Common._

lazy val root = Project("spark-knn", file(".")).
  settings(commonSettings).
  aggregate(core, examples)

lazy val core = knnProject("spark-knn-core").
  settings(Dependencies.core)

lazy val examples = knnProject("spark-knn-examples").
  dependsOn(core).
  settings(fork in run := true, coverageExcludedPackages := ".*examples.*").
  settings(Dependencies.examples).
  settings(SparkSubmit.settings: _*)
