addSbtPlugin("com.github.gseitz" % "sbt-release" % "1.0.0")

addSbtPlugin("me.lessis" % "bintray-sbt" % "0.2.1")

addSbtPlugin("com.typesafe.sbt" % "sbt-git" % "0.8.4")

addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.13.0")

addSbtPlugin("com.github.saurfang" % "sbt-spark-submit" % "0.0.2")

resolvers += Resolver.url("sbt-scoverage repo", url("https://dl.bintray.com/sksamuel/sbt-plugins"))(Resolver.ivyStylePatterns)
addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.3.0")