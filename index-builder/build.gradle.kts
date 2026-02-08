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

dependencies {
    // Lucene: core only (vectors, HNSW, IndexWriter, Document, etc.)
    // Lucene 10.x on Maven Central; use 11.x or local build for collaborative HNSW PR
    implementation("org.apache.lucene:lucene-core:10.3.2")

    // Read manifest JSON (*-meta.json)
    implementation("com.google.code.gson:gson:2.10.1")

    testImplementation("org.junit.jupiter:junit-jupiter:5.10.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

application {
    mainClass.set("lucenetestdata.indexbuilder.BuildIndex")
}

tasks.test {
    useJUnitPlatform()
}
