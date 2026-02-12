plugins {
    java
    application
}

group = "lucene-test-data"
version = "0.1.0"

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

repositories {
    mavenCentral()
}

val luceneJar: String? = findProperty("luceneJar") as String?
val luceneBackwardCodecsJar: String? = findProperty("luceneBackwardCodecsJar") as String?

dependencies {
    // Lucene: baseline = Maven 10.3.2; override with -PluceneJar=/path/to/lucene-core.jar for new-Lucene runs
    if (luceneJar != null) {
        implementation(files(luceneJar))
        if (luceneBackwardCodecsJar != null) {
            implementation(files(luceneBackwardCodecsJar))
        }
    } else {
        implementation("org.apache.lucene:lucene-core:10.3.2")
    }

    // Read manifest JSON (*-meta.json)
    implementation("com.google.code.gson:gson:2.10.1")

    testImplementation("org.junit.jupiter:junit-jupiter:5.10.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

application {
    mainClass.set("lucenetestdata.indexbuilder.BuildIndex")
}

tasks.register<JavaExec>("simpleIndexer") {
    mainClass.set("lucenetestdata.indexbuilder.SimpleIndexer")
    classpath = sourceSets["main"].runtimeClasspath
}

tasks.register<JavaExec>("mergeShards") {
    mainClass.set("lucenetestdata.indexbuilder.MergeShards")
    classpath = sourceSets["main"].runtimeClasspath
}

tasks.register<JavaExec>("runShardTest") {
    mainClass.set("lucenetestdata.indexbuilder.RunShardTest")
    classpath = sourceSets["main"].runtimeClasspath
    // Best-effort JVM settings for honest baseline (same for new-Lucene runs)
    // 20G heap for runs with --docs (2.1M vectors + 8 shards + 5k queries)
    // Vector API (SIMD) for faster KNN distance computations
    // Native access for Lucene's mmap / I/O (avoids restricted-method warnings, can help read path)
    jvmArgs(
        "-Xmx20g",
        "--add-modules", "jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED"
    )
    // Pass args: ./gradlew runShardTest --args="--shards /path --queries /path --dim 1024"
}

tasks.test {
    useJUnitPlatform()
}
