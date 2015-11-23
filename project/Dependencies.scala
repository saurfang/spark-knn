import sbt._
import Keys._

object Dependencies {
  val Versions = Seq(
    crossScalaVersions := Seq("2.10.5", "2.11.6"),
    scalaVersion := crossScalaVersions.value.head
  )

  object Compile {
    val breeze_natives = "org.scalanlp" %% "breeze-natives" % "0.11.2" % "provided"

    object Test {
      val scalatest = "org.scalatest" %% "scalatest" % "2.2.4" % "test"
      val sparktest = "org.apache.spark" %% "spark-core" % "1.5.2"  % "test" classifier "tests"
    }
  }

  import Compile._
  import Test._
  val l = libraryDependencies

  val core = l ++= Seq(scalatest, sparktest)
  val examples = core +: (l ++= Seq(breeze_natives))
}
