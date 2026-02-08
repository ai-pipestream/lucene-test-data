package lucenetestdata.indexbuilder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
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
     * @param progress      optional; called periodically during each shard write (may be null)
     */
    public static void build(DatasetResolver.ResolvedDataset resolved, Path outputPath, int numShards, int batchSize,
                            ProgressReporter progress) throws IOException {
        if (numShards < 1) {
            throw new IllegalArgumentException("numShards must be >= 1, got " + numShards);
        }
        int batch = batchSize > 0 ? batchSize : DEFAULT_BATCH_SIZE;
        EmbeddingManifest manifest = resolved.getManifest();
        int dim = manifest.getDim();
        List<float[]> vectors = VecReader.readAll(resolved.getDocsVecPath(), dim);
        if (vectors.size() != manifest.getNumDocs()) {
            throw new IllegalStateException(
                "docs.vec has " + vectors.size() + " vectors but meta.json says num_docs=" + manifest.getNumDocs());
        }
        VectorSimilarityFunction similarity = VectorSimilarityFunction.COSINE;

        if (numShards == 1) {
            Files.createDirectories(outputPath);
            writeShard(vectors, 0, vectors.size(), outputPath, similarity, batch, 0, 1, progress);
            return;
        }

        Files.createDirectories(outputPath);
        int n = vectors.size();
        int perShard = n / numShards;
        int remainder = n % numShards;
        int start = 0;
        for (int s = 0; s < numShards; s++) {
            int count = perShard + (s < remainder ? 1 : 0);
            Path shardPath = outputPath.resolve("shard-" + s);
            writeShard(vectors, start, start + count, shardPath, similarity, batch, s, numShards, progress);
            start += count;
        }
    }

    public static void build(DatasetResolver.ResolvedDataset resolved, Path outputPath, int numShards, int batchSize) throws IOException {
        build(resolved, outputPath, numShards, batchSize, null);
    }

    /** Build with default batch size. */
    public static void build(DatasetResolver.ResolvedDataset resolved, Path outputPath, int numShards) throws IOException {
        build(resolved, outputPath, numShards, 0);
    }

    /** Single-index build (numShards=1). */
    public static void build(DatasetResolver.ResolvedDataset resolved, Path indexPath) throws IOException {
        build(resolved, indexPath, 1);
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
