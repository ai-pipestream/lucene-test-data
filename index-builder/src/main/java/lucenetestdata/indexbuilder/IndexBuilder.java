package lucenetestdata.indexbuilder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.FSDirectory;

/**
 * Builds a Lucene index (or multiple shard indices) from an embedding dataset.
 * Supports pre-sharded vec files (docs-shard-{i}.vec) and concurrent shard building.
 */
public final class IndexBuilder {

    public static final String VECTOR_FIELD = "vector";
    public static final String ID_FIELD = "id";

    private static final int DEFAULT_BATCH_SIZE = 1000;
    private static final int PROGRESS_INTERVAL = 50_000;

    @FunctionalInterface
    public interface ProgressReporter {
        void report(int shardIndex, int totalShards, int docsInShard, int docsWrittenInShard);
    }

    private IndexBuilder() {}

    /**
     * Build index shards from a resolved dataset.
     * When the dataset is pre-sharded (has docs-shard-{i}.vec files), groups vec shards
     * into index shards. Otherwise reads a single docs.vec and splits in memory.
     */
    public static void build(DatasetResolver.ResolvedDataset resolved, Path outputPath,
                             int numShards, int batchSize, int numThreads,
                             ProgressReporter progress) throws IOException {
        if (numShards < 1) throw new IllegalArgumentException("numShards must be >= 1");
        int batch = batchSize > 0 ? batchSize : DEFAULT_BATCH_SIZE;
        int threads = numThreads > 0 ? numThreads : 1;
        EmbeddingManifest manifest = resolved.getManifest();
        int dim = manifest.getDim();

        Files.createDirectories(outputPath);

        if (resolved.isSharded()) {
            buildFromPreSharded(resolved, outputPath, numShards, batch, threads, dim, manifest, progress);
        } else {
            buildFromSingleFile(resolved, outputPath, numShards, batch, threads, dim, manifest, progress);
        }
    }

    /** Overload without threads (defaults to 1). */
    public static void build(DatasetResolver.ResolvedDataset resolved, Path outputPath,
                             int numShards, int batchSize, ProgressReporter progress) throws IOException {
        build(resolved, outputPath, numShards, batchSize, 1, progress);
    }

    /** Single-index build. */
    public static void build(DatasetResolver.ResolvedDataset resolved, Path indexPath) throws IOException {
        build(resolved, indexPath, 1, 0, 1, null);
    }

    private static void buildFromPreSharded(DatasetResolver.ResolvedDataset resolved, Path outputPath,
                                            int numIndexShards, int batchSize, int numThreads,
                                            int dim, EmbeddingManifest manifest,
                                            ProgressReporter progress) throws IOException {
        int numVecShards = manifest.getNumShards();
        if (numVecShards % numIndexShards != 0) {
            throw new IllegalArgumentException(
                "numIndexShards (" + numIndexShards + ") must evenly divide numVecShards (" + numVecShards + ")");
        }
        int vecShardsPerIndex = numVecShards / numIndexShards;
        List<Integer> shardDocOffsets = manifest.getShardDocOffsets();

        ExecutorService pool = Executors.newFixedThreadPool(Math.min(numThreads, numIndexShards));
        List<Future<?>> futures = new ArrayList<>();

        for (int s = 0; s < numIndexShards; s++) {
            final int shardIdx = s;
            final int firstVecShard = s * vecShardsPerIndex;
            final int globalDocOffset = shardDocOffsets.get(firstVecShard);

            futures.add(pool.submit(() -> {
                try {
                    // Read all vec shards for this index shard
                    List<float[]> vectors = new ArrayList<>();
                    for (int v = firstVecShard; v < firstVecShard + vecShardsPerIndex; v++) {
                        Path vecPath = resolved.getShardVecPath(v);
                        vectors.addAll(VecReader.readAll(vecPath, dim));
                    }

                    Path shardPath = numIndexShards == 1
                        ? outputPath
                        : outputPath.resolve("shard-" + shardIdx);

                    writeShard(vectors, 0, vectors.size(), shardPath,
                              VectorSimilarityFunction.COSINE, batchSize,
                              globalDocOffset, shardIdx, numIndexShards, progress);
                } catch (IOException e) {
                    throw new RuntimeException("Failed to build shard " + shardIdx, e);
                }
                return null;
            }));
        }

        awaitAll(futures);
        pool.shutdown();
    }

    private static void buildFromSingleFile(DatasetResolver.ResolvedDataset resolved, Path outputPath,
                                            int numShards, int batchSize, int numThreads,
                                            int dim, EmbeddingManifest manifest,
                                            ProgressReporter progress) throws IOException {
        List<float[]> vectors = VecReader.readAll(resolved.getDocsVecPath(), dim);
        if (vectors.size() != manifest.getNumDocs()) {
            throw new IllegalStateException(
                "docs.vec has " + vectors.size() + " vectors but meta.json says num_docs=" + manifest.getNumDocs());
        }

        if (numShards == 1) {
            writeShard(vectors, 0, vectors.size(), outputPath,
                       VectorSimilarityFunction.COSINE, batchSize, 0, 0, 1, progress);
            return;
        }

        int n = vectors.size();
        int perShard = n / numShards;
        int remainder = n % numShards;

        ExecutorService pool = Executors.newFixedThreadPool(Math.min(numThreads, numShards));
        List<Future<?>> futures = new ArrayList<>();
        int start = 0;

        for (int s = 0; s < numShards; s++) {
            int count = perShard + (s < remainder ? 1 : 0);
            final int from = start;
            final int to = start + count;
            final int shardIdx = s;
            Path shardPath = outputPath.resolve("shard-" + s);

            futures.add(pool.submit(() -> {
                try {
                    writeShard(vectors, from, to, shardPath,
                              VectorSimilarityFunction.COSINE, batchSize,
                              from, shardIdx, numShards, progress);
                } catch (IOException e) {
                    throw new RuntimeException("Failed to build shard " + shardIdx, e);
                }
                return null;
            }));
            start += count;
        }

        awaitAll(futures);
        pool.shutdown();
    }

    private static void writeShard(List<float[]> vectors, int from, int to, Path indexPath,
                                   VectorSimilarityFunction similarity, int batchSize,
                                   int globalDocOffset, int shardIndex, int totalShards,
                                   ProgressReporter progress) throws IOException {
        int docsInShard = to - from;
        if (progress != null) {
            progress.report(shardIndex, totalShards, docsInShard, 0);
        }
        Files.createDirectories(indexPath);
        IndexWriterConfig iwc = new IndexWriterConfig();
        iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
        try (FSDirectory dir = FSDirectory.open(indexPath);
             IndexWriter writer = new IndexWriter(dir, iwc)) {
            List<Document> batch = new ArrayList<>(Math.min(batchSize, docsInShard));
            int written = 0;
            int nextReport = Math.min(PROGRESS_INTERVAL, docsInShard);
            for (int i = from; i < to; i++) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField(VECTOR_FIELD, vectors.get(i), similarity));
                doc.add(new StoredField(ID_FIELD, globalDocOffset + (i - from)));
                batch.add(doc);
                if (batch.size() >= batchSize) {
                    writer.addDocuments(batch);
                    written += batch.size();
                    batch.clear();
                    if (progress != null && written >= nextReport) {
                        progress.report(shardIndex, totalShards, docsInShard, written);
                        nextReport = written + PROGRESS_INTERVAL;
                    }
                }
            }
            if (!batch.isEmpty()) {
                writer.addDocuments(batch);
                written += batch.size();
                if (progress != null) {
                    progress.report(shardIndex, totalShards, docsInShard, written);
                }
            }
            writer.commit();
        }
    }

    private static void awaitAll(List<Future<?>> futures) throws IOException {
        for (Future<?> f : futures) {
            try {
                f.get();
            } catch (ExecutionException e) {
                Throwable cause = e.getCause();
                if (cause instanceof IOException ioe) throw ioe;
                if (cause instanceof RuntimeException re) throw re;
                throw new RuntimeException(cause);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrupted during shard build", e);
            }
        }
    }
}
