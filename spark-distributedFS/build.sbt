name := "spark-feature-selection-relieff-cfs-svmrfe"

version := "0.1"

organization := "com.github.eirasf"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.4.0"

resolvers ++= Seq(
  "Apache Staging" at "https://repository.apache.org/content/repositories/staging/",
  "Typesafe" at "http://repo.typesafe.com/typesafe/releases",
  "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository"
)

publishMavenStyle := true

sparkPackageName := "eirasf/spark-feature-selection-relieff-cfs-svmrfe"

sparkVersion := "1.4.0"

sparkComponents += "mllib"

