package lucenetestdata.indexbuilder;

import java.nio.file.Path;
import java.util.Locale;

/**
 * CLI to build a Lucene index from an embedding dataset.
 * <p>
 * Usage:
 * <pre>
 *   java ... BuildIndex --dataset unit-data-1024-sentence --output /path/to/index
 *   java ... BuildIndex --dataset /path/to/embeddings/unit-data-1024-sentence --output /path/to/index
 * </pre>
 * If --dataset is a name (no path separators or not an existing path), it is resolved from classpath
 * (embeddings/&lt;name&gt;) or from --base if given. If --dataset is a path to a directory containing
 * meta.json and docs.vec, that directory is used.
 */
public final class BuildIndex {

    public static void main(String[] args) {
        String datasetRef = null;
        Path outputPath = null;
        Path basePath = null;
        int numShards = 1;
        int batchSize = 1000;
        int numThreads = 1;
        for (int i = 0; i < args.length; i++) {
            switch (args[i].toLowerCase(Locale.ROOT)) {
                case "--dataset", "-d" -> {
                    if (i + 1 >= args.length) usage("Missing value for --dataset");
                    datasetRef = args[++i];
                }
                case "--output", "-o" -> {
                    if (i + 1 >= args.length) usage("Missing value for --output");
                    outputPath = Path.of(args[++i]);
                }
                case "--base" -> {
                    if (i + 1 >= args.length) usage("Missing value for --base");
                    basePath = Path.of(args[++i]);
                }
                case "--num-shards", "-n" -> {
                    if (i + 1 >= args.length) usage("Missing value for --num-shards");
                    numShards = Integer.parseInt(args[++i]);
                    if (numShards < 1) usage("--num-shards must be >= 1");
                }
                case "--batch-size", "-b" -> {
                    if (i + 1 >= args.length) usage("Missing value for --batch-size");
                    batchSize = Integer.parseInt(args[++i]);
                    if (batchSize < 1) usage("--batch-size must be >= 1");
                }
                case "--threads", "-t" -> {
                    if (i + 1 >= args.length) usage("Missing value for --threads");
                    numThreads = Integer.parseInt(args[++i]);
                    if (numThreads < 1) usage("--threads must be >= 1");
                }
                case "--help", "-h" -> usage(null);
                default -> usage("Unknown option: " + args[i]);
            }
        }
        if (datasetRef == null) usage("--dataset is required");
        if (outputPath == null) usage("--output is required");
        try {
            DatasetResolver.ResolvedDataset resolved = DatasetResolver.resolve(datasetRef, basePath);
            System.out.println("Dataset: " + resolved.getDatasetDir());
            System.out.println("  dim=" + resolved.getManifest().getDim()
                + " num_docs=" + resolved.getManifest().getNumDocs());
            if (numShards > 1) {
                System.out.println("Building " + numShards + " shard indices under: " + outputPath
                    + " (batch=" + batchSize + ", threads=" + numThreads + ")");
            } else {
                System.out.println("Building index at: " + outputPath + " (batch=" + batchSize + ")");
            }
            if (resolved.isSharded()) {
                System.out.println("  Pre-sharded dataset: " + resolved.getManifest().getNumShards() + " vec shards");
            }
            IndexBuilder.ProgressReporter progress = (shard, total, inShard, written) -> {
                if (total > 1) {
                    System.err.printf("  Shard %d/%d: %d/%d docs%n", shard + 1, total, written, inShard);
                }
            };
            IndexBuilder.build(resolved, outputPath, numShards, batchSize, numThreads, progress);
            System.out.println("Done.");
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void usage(String error) {
        if (error != null) System.err.println(error);
        System.err.println("Usage: BuildIndex --dataset <name|path> --output <index-dir> [options]");
        System.err.println("  --dataset     Dataset name (e.g. unit-data-1024-sentence) or path to dataset dir");
        System.err.println("  --output      Lucene index output directory (or base dir when --num-shards > 1)");
        System.err.println("  --num-shards  Number of shard indices to create (default: 1). Output: <output>/shard-0, shard-1, ...");
        System.err.println("  --batch-size  Documents per addDocuments batch (default: 1000). Larger = faster indexing, more RAM.");
        System.err.println("  --threads     Number of concurrent threads for shard building (default: 1).");
        System.err.println("  --base        Base directory containing an 'embeddings' subdir (e.g. data or repo root)");
        System.exit(error == null ? 0 : 1);
    }
}
