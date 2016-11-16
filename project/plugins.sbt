addSbtPlugin("com.github.gseitz" % "sbt-release" % "1.0.0")

addSbtPlugin("me.lessis" % "bintray-sbt" % "0.2.1")

addSbtPlugin("com.typesafe.sbt" % "sbt-git" % "0.8.4")

addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.13.0")

addSbtPlugin("com.github.saurfang" % "sbt-spark-submit" % "0.0.2")

addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.5.0")

addSbtPlugin("org.scalastyle" %% "scalastyle-sbt-plugin" % "0.7.0"
  excludeAll ExclusionRule(organization = "com.danieltrinh"))
libraryDependencies += "org.scalariform" %% "scalariform" % "0.1.7"

resolvers += "bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven/"
addSbtPlugin("org.spark-packages" % "sbt-spark-package" % "0.2.5")
