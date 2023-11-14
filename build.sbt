import Common._

lazy val root = Project("spark-knn", file(".")).
  settings(commonSettings).
  settings(Dependencies.Versions).
  aggregate(core, examples)

lazy val core = knnProject("spark-knn-core").
  settings(
    name := "spark-knn",
    credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials"),
    licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")
  ).
  settings(Dependencies.core).
  settings(
    ThisBuild / scalafixDependencies += "org.scalatest" %% "autofix" % "3.1.0.1",
    addCompilerPlugin(scalafixSemanticdb) // enable SemanticDB
  )

lazy val examples = knnProject("spark-knn-examples").
  dependsOn(core).
  settings(fork in run := true, coverageExcludedPackages := ".*examples.*").
  settings(Dependencies.examples)
