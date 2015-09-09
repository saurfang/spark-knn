import Common._

lazy val root = Project("spark-knn", file(".")).
  settings(commonSettings: _*).
  aggregate(core, examples)

lazy val core = tsneProject("spark-knn-core").
  settings(Dependencies.core).
  settings(coverageEnabled := true)

lazy val examples = tsneProject("spark-knn-examples").
  dependsOn(core).
  settings(fork in run := true).
  settings(Dependencies.examples).
  settings(SparkSubmit.settings: _*)
