import com.typesafe.sbt.GitVersioning
import sbt._
import Keys._
import com.typesafe.sbt.GitPlugin.autoImport._
import sbtsparkpackage.SparkPackagePlugin.autoImport._

import scala.language.experimental.macros
import scala.reflect.macros.Context

object Common {
  val commonSettings = Seq(
    organization in ThisBuild := "com.github.saurfang",
    javacOptions ++= Seq("-source", "1.7", "-target", "1.7"),
    scalacOptions ++= Seq("-target:jvm-1.7", "-deprecation", "-feature"),
    //git.useGitDescribe := true,
    git.baseVersion := "0.0.1",
    parallelExecution in test := false,
    updateOptions := updateOptions.value.withCachedResolution(true),
    sparkVersion := "2.0.0",
    sparkComponents += "mllib",
    spIgnoreProvided := true
  )

  def knnProject(path: String): Project = macro knnProjectMacroImpl

  def knnProjectMacroImpl(c: Context)(path: c.Expr[String]) = {
    import c.universe._
    reify {
      (Project.projectMacroImpl(c).splice in file(path.splice)).
        enablePlugins(GitVersioning).
        settings(name := path.splice).
        settings(Dependencies.Versions).
        settings(commonSettings)
    }
  }
}
