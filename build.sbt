ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.7.4"

lazy val root = (project in file("."))
  .settings(
    name := "javaTorchs"
  )
resolvers += "Central Portal Snapshots" at "https://central.sonatype.com/repository/maven-snapshots/"
resolvers += "aliyun" at "https://maven.aliyun.com/repository/public"
// 依赖项
libraryDependencies ++= Seq(
  "org.bytedeco" % "pytorch-platform" % "2.9.1-1.5.13-SNAPSHOT",
  //  "org.bytedeco" % "pytorch-platform-gpu" % "2.9.1-1.5.13-SNAPSHOT",
  "org.bytedeco" % "cuda-platform-redist-cudnn" % "13.0-9.14-1.5.13-SNAPSHOT",
  "org.bytedeco" % "cuda-platform-redist-cusolver" % "13.0-9.14-1.5.13-SNAPSHOT",
  "org.bytedeco" % "cuda-platform-redist-nccl" % "13.0-9.14-1.5.13-SNAPSHOT"
  // 注释掉的 MKL 依赖
  // "org.bytedeco" % "mkl-platform-redist" % "2025.2-1.5.13-SNAPSHOT"
)