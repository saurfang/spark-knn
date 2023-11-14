import sbt._
import Keys._

object Dependencies {
  val Versions = Seq(
    crossScalaVersions := Seq("2.12.18", "2.13.12"),
    scalaVersion := crossScalaVersions.value.head
  )

  object Compile {
    val spark_version = "3.4.1"
    val spark_core = "org.apache.spark" %% "spark-core" % spark_version % "provided"
    val spark_mllib = "org.apache.spark" %% "spark-mllib" % spark_version % "provided"
    val breeze = "org.scalanlp" %% "breeze" % "2.1.0" % "provided"
    val netlib = "com.github.fommil.netlib" % "core" % "1.1.2"

    object Test {
      val scalatest = "org.scalatest" %% "scalatest" % "3.2.17" % "test"
      val sparktest = "org.apache.spark" %% "spark-core" % spark_version  % "test" classifier "tests"
    }
  }

  import Compile._
  import Test._
  val l = libraryDependencies

  val core = l ++= Seq(spark_core, spark_mllib, scalatest, sparktest)
  val examples = core +: (l ++= Seq(breeze, netlib))
}
