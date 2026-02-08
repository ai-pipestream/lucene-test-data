package lucenetestdata.indexbuilder;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Functional tests using embedded unit-data datasets (from resources).
 * Uses --base build/resources/main when running under Gradle.
 */
class IndexBuilderTest {

    private static Path getResourcesBase() {
        Path buildRes = Paths.get("build/resources/main");
        if (Files.isDirectory(buildRes)) return buildRes.toAbsolutePath();
        return Paths.get("src/main/resources").toAbsolutePath();
    }

    @Test
    void resolveDatasetFromBase() throws Exception {
        Path base = getResourcesBase();
        DatasetResolver.ResolvedDataset resolved = DatasetResolver.resolve("unit-data-1024-sentence", base);
        assertNotNull(resolved);
        assertNotNull(resolved.getManifest());
        assertEquals(1024, resolved.getManifest().getDim());
        assertTrue(resolved.getManifest().getNumDocs() > 0);
        assertTrue(Files.isRegularFile(resolved.getDocsVecPath()));
        assertTrue(Files.isRegularFile(resolved.getQueriesVecPath()));
    }

    @Test
    void buildIndexAndSearch(@TempDir Path tmp) throws Exception {
        Path base = getResourcesBase();
        DatasetResolver.ResolvedDataset resolved = DatasetResolver.resolve("unit-data-1024-sentence", base);
        Path indexPath = tmp.resolve("index");
        IndexBuilder.build(resolved, indexPath);

        try (Directory dir = FSDirectory.open(indexPath);
             DirectoryReader reader = DirectoryReader.open(dir)) {
            assertEquals(resolved.getManifest().getNumDocs(), reader.numDocs());
            IndexSearcher searcher = new IndexSearcher(reader);
            // Use first query vector from dataset as query (we'd load queries.vec in real use)
            float[] queryVec = VecReader.readAll(resolved.getQueriesVecPath(), resolved.getManifest().getDim()).get(0);
            var query = new KnnFloatVectorQuery(IndexBuilder.VECTOR_FIELD, queryVec, 10);
            var topDocs = searcher.search(query, 10);
            assertTrue(topDocs.scoreDocs.length > 0);
        }
    }

    @Test
    void resolveByPathWhenAbsolute(@TempDir Path tmp) throws Exception {
        Path base = getResourcesBase();
        DatasetResolver.ResolvedDataset fromBase = DatasetResolver.resolve("unit-data-1024-sentence", base);
        assertNotNull(fromBase);
        Path copyDir = tmp.resolve("copy");
        Files.createDirectories(copyDir);
        Files.copy(fromBase.getDatasetDir().resolve("meta.json"), copyDir.resolve("meta.json"));
        Files.copy(fromBase.getDatasetDir().resolve("docs.vec"), copyDir.resolve("docs.vec"));
        Files.copy(fromBase.getDatasetDir().resolve("queries.vec"), copyDir.resolve("queries.vec"));

        DatasetResolver.ResolvedDataset byPath = DatasetResolver.resolve(copyDir.toAbsolutePath().toString(), null);
        assertNotNull(byPath);
        assertEquals(fromBase.getManifest().getDim(), byPath.getManifest().getDim());
        assertEquals(fromBase.getManifest().getNumDocs(), byPath.getManifest().getNumDocs());
    }

    @Test
    void buildMultipleShards(@TempDir Path tmp) throws Exception {
        Path base = getResourcesBase();
        DatasetResolver.ResolvedDataset resolved = DatasetResolver.resolve("unit-data-1024-sentence", base);
        Path outputBase = tmp.resolve("index");
        int numShards = 4;
        IndexBuilder.build(resolved, outputBase, numShards);

        int totalDocs = 0;
        for (int s = 0; s < numShards; s++) {
            Path shardPath = outputBase.resolve("shard-" + s);
            assertTrue(Files.isDirectory(shardPath), "shard dir exists: " + shardPath);
            try (Directory dir = FSDirectory.open(shardPath);
                 DirectoryReader reader = DirectoryReader.open(dir)) {
                totalDocs += reader.numDocs();
            }
        }
        assertEquals(resolved.getManifest().getNumDocs(), totalDocs);
    }
}
