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
 * Builds a Lucene index (or multiple shard indices) from an embedding dataset (docs.vec + meta.json).
 * Uses batched addDocuments for faster indexing.
 */
public final class IndexBuilder {

    public static final String VECTOR_FIELD = "vector";
    /** Global document id (index in original docs.vec); stored for recall/merge across shards. */
    public static final String ID_FIELD = "id";

    private static final int DEFAULT_BATCH_SIZE = 1000;
    /** Report progress every this many documents per shard. */
    private static final int PROGRESS_INTERVAL = 50_000;

    /** Called with (shardIndex, totalShards, docsInShard, docsWrittenInShard). */
    @FunctionalInterface
    public interface ProgressReporter {
        void report(int shardIndex, int totalShards, int docsInShard, int docsWrittenInShard);
    }

    private IndexBuilder() {}

    /**
     * Build one or more indices from a resolved dataset.
     *
     * @param resolved      dataset with manifest and paths
     * @param outputPath    directory for the index (when numShards==1) or base dir for shard subdirs
     * @param numShards     number of shards (indices) to create; docs are split contiguously by global id
     * @param batchSize     number of documents per addDocuments batch (e.g. 1000); use 0 for default
     * @param numThreads    number of threads for concurrent shard building; use 1 for sequential
     * @param progress      optional; called periodically during each shard write (may be null).
     *                      Must be thread-safe when numThreads > 1.
     */
    public static void build(DatasetResolver.ResolvedDataset resolved, Path outputPath, int numShards, int batchSize,
                            int numThreads, ProgressReporter progress) throws IOException {
        if (numShards < 1) {
            throw new IllegalArgumentException("numShards must be >= 1, got " + numShards);
        }
        int batch = batchSize > 0 ? batchSize : DEFAULT_BATCH_SIZE;
        int threads = numThreads > 0 ? numThreads : 1;
        EmbeddingManifest manifest = resolved.getManifest();
        int dim = manifest.getDim();
        VectorSimilarityFunction similarity = VectorSimilarityFunction.COSINE;

        // Pre-sharded dataset: group vec shards into index shards.
        // E.g. 16 vec shards with numShards=4 → each index shard reads 4 contiguous vec files.
        // numShards=1 → all vec shards merge into one index.
        if (resolved.isSharded()) {
            int numVecShards = manifest.getNumShards();
            List<Integer> shardSizes = manifest.getShardSizes();
            List<Integer> shardOffsets = manifest.getShardDocOffsets();
            if (numVecShards % numShards != 0) {
                throw new IllegalArgumentException(
                    "numShards (" + numShards + ") must evenly divide the number of vec shards ("
                    + numVecShards + ")");
            }
            int vecShardsPerIndex = numVecShards / numShards;
            Files.createDirectories(outputPath);

            if (numShards == 1 || threads <= 1) {
                // Sequential
                for (int s = 0; s < numShards; s++) {
                    buildShardedIndex(resolved, s, vecShardsPerIndex, shardSizes, shardOffsets,
                            dim, similarity, batch, numShards, outputPath, progress);
                }
            } else {
                // Concurrent
                int poolSize = Math.min(threads, numShards);
                ExecutorService pool = Executors.newFixedThreadPool(poolSize);
                List<Future<?>> futures = new ArrayList<>(numShards);
                for (int s = 0; s < numShards; s++) {
                    final int shard = s;
                    futures.add(pool.submit(() -> {
                        buildShardedIndex(resolved, shard, vecShardsPerIndex, shardSizes, shardOffsets,
                                dim, similarity, batch, numShards, outputPath, progress);
                        return null;
                    }));
                }
                pool.shutdown();
                awaitAll(futures);
            }
            return;
        }

        // Unsharded: load all vectors, optionally split into index shards
        List<float[]> vectors = VecReader.readAll(resolved.getDocsVecPath(), dim);
        if (vectors.size() != manifest.getNumDocs()) {
            throw new IllegalStateException(
                "docs.vec has " + vectors.size() + " vectors but meta.json says num_docs=" + manifest.getNumDocs());
        }

        if (numShards == 1) {
            Files.createDirectories(outputPath);
            writeShard(vectors, 0, vectors.size(), outputPath, similarity, batch, 0, 1, progress);
            return;
        }

        Files.createDirectories(outputPath);
        int n = vectors.size();
        int perShard = n / numShards;
        int remainder = n % numShards;

        // Build shard ranges
        int[][] ranges = new int[numShards][2];
        int start = 0;
        for (int s = 0; s < numShards; s++) {
            int count = perShard + (s < remainder ? 1 : 0);
            ranges[s] = new int[]{start, start + count};
            start += count;
        }

        if (threads <= 1) {
            for (int s = 0; s < numShards; s++) {
                Path shardPath = outputPath.resolve("shard-" + s);
                writeShard(vectors, ranges[s][0], ranges[s][1], shardPath, similarity, batch, s, numShards, progress);
            }
        } else {
            int poolSize = Math.min(threads, numShards);
            ExecutorService pool = Executors.newFixedThreadPool(poolSize);
            List<Future<?>> futures = new ArrayList<>(numShards);
            for (int s = 0; s < numShards; s++) {
                final int shard = s;
                futures.add(pool.submit(() -> {
                    Path shardPath = outputPath.resolve("shard-" + shard);
                    writeShard(vectors, ranges[shard][0], ranges[shard][1], shardPath, similarity, batch,
                            shard, numShards, progress);
                    return null;
                }));
            }
            pool.shutdown();
            awaitAll(futures);
        }
    }

    public static void build(DatasetResolver.ResolvedDataset resolved, Path outputPath, int numShards, int batchSize,
                            ProgressReporter progress) throws IOException {
        build(resolved, outputPath, numShards, batchSize, 1, progress);
    }

    public static void build(DatasetResolver.ResolvedDataset resolved, Path outputPath, int numShards, int batchSize) throws IOException {
        build(resolved, outputPath, numShards, batchSize, 1, null);
    }

    /** Build with default batch size. */
    public static void build(DatasetResolver.ResolvedDataset resolved, Path outputPath, int numShards) throws IOException {
        build(resolved, outputPath, numShards, 0);
    }

    /** Single-index build (numShards=1). */
    public static void build(DatasetResolver.ResolvedDataset resolved, Path indexPath) throws IOException {
        build(resolved, indexPath, 1);
    }

    /** Load vec shard files for one index shard and write the Lucene index. */
    private static void buildShardedIndex(DatasetResolver.ResolvedDataset resolved, int indexShard,
                                           int vecShardsPerIndex, List<Integer> shardSizes,
                                           List<Integer> shardOffsets, int dim,
                                           VectorSimilarityFunction similarity, int batchSize,
                                           int numShards, Path outputPath,
                                           ProgressReporter progress) throws IOException {
        int firstVec = indexShard * vecShardsPerIndex;
        List<float[]> combined = new ArrayList<>();
        for (int v = firstVec; v < firstVec + vecShardsPerIndex; v++) {
            Path shardVecPath = resolved.getShardVecPath(v);
            List<float[]> vecChunk = VecReader.readAll(shardVecPath, dim);
            if (vecChunk.size() != shardSizes.get(v)) {
                throw new IllegalStateException(
                    shardVecPath.getFileName() + " has " + vecChunk.size()
                    + " vectors but meta.json says shard_sizes[" + v + "]=" + shardSizes.get(v));
            }
            combined.addAll(vecChunk);
        }
        int globalOffset = shardOffsets.get(firstVec);
        Path shardPath = numShards == 1 ? outputPath : outputPath.resolve("shard-" + indexShard);
        writeShardWithOffset(combined, shardPath, similarity, batchSize, globalOffset, indexShard, numShards, progress);
    }

    /** Wait for all futures, unwrapping IOException from ExecutionException. */
    private static void awaitAll(List<Future<?>> futures) throws IOException {
        for (Future<?> f : futures) {
            try {
                f.get();
            } catch (ExecutionException e) {
                Throwable cause = e.getCause();
                if (cause instanceof IOException ioe) throw ioe;
                if (cause instanceof RuntimeException re) throw re;
                throw new IOException("Shard build failed", cause);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrupted while building shards", e);
            }
        }
    }

    /**
     * Write an index shard from a pre-sharded vec file, using globalOffset to compute global doc IDs.
     */
    private static void writeShardWithOffset(List<float[]> vectors, Path indexPath,
                                              VectorSimilarityFunction similarity, int batchSize,
                                              int globalOffset, int shardIndex, int totalShards,
                                              ProgressReporter progress) throws IOException {
        int docsInShard = vectors.size();
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
            for (int i = 0; i < docsInShard; i++) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField(VECTOR_FIELD, vectors.get(i), similarity));
                doc.add(new StoredField(ID_FIELD, globalOffset + i));
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

    private static void writeShard(List<float[]> vectors, int from, int to, Path indexPath,
                                   VectorSimilarityFunction similarity, int batchSize,
                                   int shardIndex, int totalShards, ProgressReporter progress) throws IOException {
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
                doc.add(new StoredField(ID_FIELD, i));
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
}
