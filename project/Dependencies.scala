import sbt._
import Keys._

object Dependencies {
  val Versions = Seq(
    crossScalaVersions := Seq("2.10.5", "2.11.6"),
    scalaVersion := crossScalaVersions.value.head
  )

  object Compile {
    val spark = "org.apache.spark" %% "spark-mllib" % "1.5.0" % "provided"
    val breeze_natives = "org.scalanlp" %% "breeze-natives" % "0.11.2" % "provided"

    object Test {
      val scalatest = "org.scalatest" %% "scalatest" % "2.2.4" % "test"
    }
  }

  import Compile._
  val l = libraryDependencies

  val core = l ++= Seq(spark, Test.scalatest)
  val examples = core +: (l ++= Seq(breeze_natives))
}