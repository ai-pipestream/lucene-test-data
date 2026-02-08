package lucenetestdata.indexbuilder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.FSDirectory;

/**
 * Merge groups of Lucene shard indices into fewer, larger shards.
 * Uses IndexWriter.addIndexes() which copies segments directly — no re-indexing needed.
 * <p>
 * Example: merge 16 shards into 4 by grouping every 4 consecutive shards:
 * <pre>
 *   java ... MergeShards --input data/indices/wiki-1024-sentences/shards-16 \
 *       --output data/indices/wiki-1024-sentences/shards-4 \
 *       --source-shards 16 --target-shards 4 --threads 4
 * </pre>
 */
public final class MergeShards {

    public static void main(String[] args) {
        Path inputPath = null;
        Path outputPath = null;
        int sourceShards = 0;
        int targetShards = 0;
        int numThreads = 1;
        for (int i = 0; i < args.length; i++) {
            switch (args[i].toLowerCase(Locale.ROOT)) {
                case "--input", "-i" -> {
                    if (i + 1 >= args.length) usage("Missing value for --input");
                    inputPath = Path.of(args[++i]);
                }
                case "--output", "-o" -> {
                    if (i + 1 >= args.length) usage("Missing value for --output");
                    outputPath = Path.of(args[++i]);
                }
                case "--source-shards" -> {
                    if (i + 1 >= args.length) usage("Missing value for --source-shards");
                    sourceShards = Integer.parseInt(args[++i]);
                }
                case "--target-shards" -> {
                    if (i + 1 >= args.length) usage("Missing value for --target-shards");
                    targetShards = Integer.parseInt(args[++i]);
                }
                case "--threads", "-t" -> {
                    if (i + 1 >= args.length) usage("Missing value for --threads");
                    numThreads = Integer.parseInt(args[++i]);
                }
                case "--help", "-h" -> usage(null);
                default -> usage("Unknown option: " + args[i]);
            }
        }
        if (inputPath == null) usage("--input is required");
        if (outputPath == null) usage("--output is required");
        if (sourceShards < 1) usage("--source-shards is required and must be >= 1");
        if (targetShards < 1) usage("--target-shards is required and must be >= 1");
        if (sourceShards % targetShards != 0) {
            usage("--source-shards (" + sourceShards + ") must be evenly divisible by --target-shards (" + targetShards + ")");
        }
        if (targetShards >= sourceShards) {
            usage("--target-shards must be less than --source-shards");
        }

        try {
            merge(inputPath, outputPath, sourceShards, targetShards, numThreads);
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Merge source shard indices into fewer target shards using addIndexes().
     *
     * @param inputBase    directory containing shard-0, shard-1, ... shard-(sourceShards-1)
     * @param outputBase   directory to write merged shard-0, shard-1, ... shard-(targetShards-1)
     * @param sourceShards number of existing shards
     * @param targetShards number of merged shards to produce (must evenly divide sourceShards)
     * @param numThreads   concurrent merge threads
     */
    public static void merge(Path inputBase, Path outputBase, int sourceShards, int targetShards,
                              int numThreads) throws IOException {
        int groupSize = sourceShards / targetShards;
        System.out.println("Merging " + sourceShards + " shards → " + targetShards
            + " shards (groups of " + groupSize + ", threads=" + numThreads + ")");
        System.out.println("  Input:  " + inputBase);
        System.out.println("  Output: " + outputBase);

        Files.createDirectories(outputBase);
        int threads = Math.min(numThreads, targetShards);

        if (threads <= 1) {
            for (int t = 0; t < targetShards; t++) {
                mergeGroup(inputBase, outputBase, t, groupSize);
            }
        } else {
            ExecutorService pool = Executors.newFixedThreadPool(threads);
            List<Future<?>> futures = new ArrayList<>(targetShards);
            for (int t = 0; t < targetShards; t++) {
                final int target = t;
                futures.add(pool.submit(() -> {
                    mergeGroup(inputBase, outputBase, target, groupSize);
                    return null;
                }));
            }
            pool.shutdown();
            for (Future<?> f : futures) {
                try {
                    f.get();
                } catch (ExecutionException e) {
                    Throwable cause = e.getCause();
                    if (cause instanceof IOException ioe) throw ioe;
                    if (cause instanceof RuntimeException re) throw re;
                    throw new IOException("Merge failed", cause);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IOException("Interrupted during merge", e);
                }
            }
        }
        System.out.println("Done. Created " + targetShards + " merged shards under " + outputBase);
    }

    private static void mergeGroup(Path inputBase, Path outputBase, int targetIndex, int groupSize) throws IOException {
        int firstSource = targetIndex * groupSize;
        Path targetPath = outputBase.resolve("shard-" + targetIndex);
        Files.createDirectories(targetPath);

        System.out.println("  shard-" + targetIndex + " ← source shards " + firstSource + ".." + (firstSource + groupSize - 1));

        IndexWriterConfig iwc = new IndexWriterConfig();
        iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
        try (FSDirectory targetDir = FSDirectory.open(targetPath);
             IndexWriter writer = new IndexWriter(targetDir, iwc)) {
            for (int s = firstSource; s < firstSource + groupSize; s++) {
                Path sourcePath = inputBase.resolve("shard-" + s);
                try (FSDirectory sourceDir = FSDirectory.open(sourcePath)) {
                    writer.addIndexes(sourceDir);
                }
            }
            writer.commit();
        }
    }

    private static void usage(String error) {
        if (error != null) System.err.println(error);
        System.err.println("Usage: MergeShards --input <shards-dir> --output <merged-dir> --source-shards N --target-shards M [options]");
        System.err.println("  --input          Directory containing shard-0 .. shard-(N-1) index dirs");
        System.err.println("  --output         Directory to write merged shard-0 .. shard-(M-1)");
        System.err.println("  --source-shards  Number of existing shards (N)");
        System.err.println("  --target-shards  Number of merged shards to produce (M); N must be divisible by M");
        System.err.println("  --threads        Concurrent merge threads (default: 1)");
        System.exit(error == null ? 0 : 1);
    }
}
