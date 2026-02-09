package lucenetestdata.indexbuilder;

import com.google.gson.Gson;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Resolves a dataset (by name or path) and loads its manifest.
 * Dataset dir must contain docs.vec and meta.json.
 */
public final class DatasetResolver {

    private static final String EMBEDDINGS_RESOURCE_PREFIX = "embeddings/";
    private static final Gson GSON = new Gson();

    private DatasetResolver() {}

    /**
     * Resolve dataset directory and load manifest.
     *
     * @param datasetRef either a dataset name (e.g. "unit-data-1024-sentence") or a path to the dataset directory
     * @param basePath   optional base path for file lookup when datasetRef is a name; if null, classpath is used
     * @return manifest and the resolved dataset directory path
     */
    public static ResolvedDataset resolve(String datasetRef, Path basePath) throws IOException {
        Path datasetDir;
        Path asPath = Path.of(datasetRef);
        if (asPath.isAbsolute() && Files.isDirectory(asPath)) {
            datasetDir = asPath;
        } else if (basePath != null && Files.isDirectory(basePath)) {
            Path candidate = basePath.resolve("embeddings").resolve(datasetRef);
            if (Files.isDirectory(candidate)) {
                datasetDir = candidate;
            } else {
                candidate = basePath.resolve(datasetRef);
                datasetDir = Files.isDirectory(candidate) ? candidate : null;
            }
        } else {
            datasetDir = null;
        }
        if (datasetDir == null) {
            datasetDir = resolveFromClasspath(datasetRef);
        }
        if (datasetDir == null || !Files.isDirectory(datasetDir)) {
            throw new IllegalArgumentException("Dataset not found: " + datasetRef);
        }
        Path metaPath = datasetDir.resolve("meta.json");
        if (!Files.isRegularFile(metaPath)) {
            throw new IllegalArgumentException("Missing meta.json in dataset dir: " + datasetDir);
        }
        EmbeddingManifest manifest = loadManifest(metaPath);
        // Sharded datasets have docs-shard-0.vec; unsharded have docs.vec
        if (!manifest.isSharded()) {
            Path docsPath = datasetDir.resolve("docs.vec");
            if (!Files.isRegularFile(docsPath)) {
                throw new IllegalArgumentException("Missing docs.vec in dataset dir: " + datasetDir);
            }
        } else {
            Path shard0 = datasetDir.resolve("docs-shard-0.vec");
            if (!Files.isRegularFile(shard0)) {
                throw new IllegalArgumentException("Missing docs-shard-0.vec in sharded dataset dir: " + datasetDir);
            }
        }
        return new ResolvedDataset(datasetDir, manifest);
    }

    private static Path resolveFromClasspath(String name) {
        String resourcePath = EMBEDDINGS_RESOURCE_PREFIX + name + "/meta.json";
        try (InputStream in = DatasetResolver.class.getClassLoader().getResourceAsStream(resourcePath)) {
            if (in == null) return null;
            // Resolve directory from resource URL when running from filesystem (e.g. IDE or exploded JAR)
            java.net.URL url = DatasetResolver.class.getClassLoader().getResource(resourcePath);
            if (url != null && "file".equals(url.getProtocol())) {
                try {
                    Path metaFile = Path.of(url.toURI());
                    return metaFile.getParent();
                } catch (URISyntaxException e) {
                    return null;
                }
            }
        } catch (IOException e) {
            return null;
        }
        return null;
    }

    public static EmbeddingManifest loadManifest(Path metaPath) throws IOException {
        try (Reader r = Files.newBufferedReader(metaPath, StandardCharsets.UTF_8)) {
            return GSON.fromJson(r, EmbeddingManifest.class);
        }
    }

    public static final class ResolvedDataset {
        private final Path datasetDir;
        private final EmbeddingManifest manifest;

        ResolvedDataset(Path datasetDir, EmbeddingManifest manifest) {
            this.datasetDir = datasetDir;
            this.manifest = Objects.requireNonNull(manifest);
        }

        public Path getDatasetDir() { return datasetDir; }
        public EmbeddingManifest getManifest() { return manifest; }
        public Path getDocsVecPath() { return datasetDir.resolve("docs.vec"); }
        public Path getShardVecPath(int shardIndex) { return datasetDir.resolve("docs-shard-" + shardIndex + ".vec"); }
        public boolean isSharded() { return manifest.isSharded(); }
        public Path getQueriesVecPath() { return datasetDir.resolve("queries.vec"); }
    }
}
