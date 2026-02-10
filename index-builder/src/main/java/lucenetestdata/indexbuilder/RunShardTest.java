package lucenetestdata.indexbuilder;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Runs KNN tests across existing shard indices: K=10,100,1000,10000.
 * No index creation; loads shards from a directory (shard-0, shard-1, ...), loads queries,
 * optionally docs for recall. Prints luceneutil-style SUMMARY and table.
 * <p>
 * Usage:
 * <pre>
 *   RunShardTest --shards /path/to/index-simplewiki-8shards --queries /path/to/queries.vec --dim 1024 [--docs /path/to/docs.vec] [--k 10,100,1000,10000]
 * </pre>
 */
public final class RunShardTest {

    private static final Pattern SHARD_DIR = Pattern.compile("shard-(\\d+)");

    public static void main(String[] args) {
        Path shardsDir = null;
        Path queriesPath = null;
        Path docsPath = null;
        int dim = 0;
        int searchThreads = 0; // 0 = default
        int queryThreads = 1;  // parallel queries (1 = sequential)
        int maxQueries = 0;    // 0 = all
        int debugQueries = 0;  // 0 = off; print per-query overlap for first N
        Path dumpTopKPath = null;
        boolean collaborative = false;
        boolean isolated = false;
        boolean progress = false;
        int[] kValues = { 10, 100, 1000, 10_000 };
        for (int i = 0; i < args.length; i++) {
            switch (args[i].toLowerCase(Locale.ROOT)) {
                case "--collaborative" -> collaborative = true;
                case "--isolated" -> isolated = true;
                case "--progress" -> progress = true;
                case "--shards", "-s" -> {
                    if (i + 1 >= args.length) usage("Missing value for --shards");
                    shardsDir = Path.of(args[++i]);
                }
                case "--queries", "-q" -> {
                    if (i + 1 >= args.length) usage("Missing value for --queries");
                    queriesPath = Path.of(args[++i]);
                }
                case "--docs", "-d" -> {
                    if (i + 1 >= args.length) usage("Missing value for --docs");
                    docsPath = Path.of(args[++i]);
                }
                case "--dim" -> {
                    if (i + 1 >= args.length) usage("Missing value for --dim");
                    dim = Integer.parseInt(args[++i]);
                    if (dim < 1) usage("--dim must be >= 1");
                }
                case "--k" -> {
                    if (i + 1 >= args.length) usage("Missing value for --k");
                    String[] parts = args[++i].split(",");
                    kValues = new int[parts.length];
                    for (int j = 0; j < parts.length; j++) {
                        kValues[j] = Integer.parseInt(parts[j].trim());
                        if (kValues[j] < 1) usage("--k values must be >= 1");
                    }
                }
                case "--search-threads" -> {
                    if (i + 1 >= args.length) usage("Missing value for --search-threads");
                    searchThreads = Integer.parseInt(args[++i]);
                    if (searchThreads < 1) usage("--search-threads must be >= 1");
                }
                case "--query-threads" -> {
                    if (i + 1 >= args.length) usage("Missing value for --query-threads");
                    queryThreads = Integer.parseInt(args[++i]);
                    if (queryThreads < 1) usage("--query-threads must be >= 1");
                }
                case "--max-queries" -> {
                    if (i + 1 >= args.length) usage("Missing value for --max-queries");
                    maxQueries = Integer.parseInt(args[++i]);
                    if (maxQueries < 1) usage("--max-queries must be >= 1");
                }
                case "--debug-queries" -> {
                    if (i + 1 >= args.length) usage("Missing value for --debug-queries");
                    debugQueries = Integer.parseInt(args[++i]);
                    if (debugQueries < 1) usage("--debug-queries must be >= 1");
                }
                case "--dump-topk" -> {
                    if (i + 1 >= args.length) usage("Missing value for --dump-topk");
                    dumpTopKPath = Path.of(args[++i]);
                }
                case "--help", "-h" -> usage(null);
                default -> usage("Unknown option: " + args[i]);
            }
        }
        if (shardsDir == null) usage("--shards is required");
        if (queriesPath == null) usage("--queries is required");
        if (dim <= 0) usage("--dim is required");
        if (isolated && !collaborative) {
            usage("--isolated requires --collaborative");
        }

        List<Path> shardPaths = List.of();
        try {
            shardPaths = discoverShards(shardsDir);
        } catch (IOException e) {
            System.err.println("Failed to list shards: " + e.getMessage());
            System.exit(1);
        }
        if (shardPaths.isEmpty()) {
            System.err.println("No shard-* directories found under " + shardsDir);
            System.exit(1);
        }

        int numShardsForThreads = shardPaths.size();
        int effectiveSearchThreads = searchThreads > 0
            ? searchThreads
            : (queryThreads > 1 ? numShardsForThreads * queryThreads : numShardsForThreads);
        System.out.println("Shards: " + shardsDir + " (" + shardPaths.size() + " shards)");
        System.out.println("Search threads: " + effectiveSearchThreads + (searchThreads > 0 ? "" : " (default)"));
        System.out.println("Query threads: " + queryThreads + (queryThreads > 1 ? " (parallel queries)" : ""));
        System.out.println("Queries: " + queriesPath + " dim=" + dim);
        if (maxQueries > 0) {
            System.out.println("Max queries: " + maxQueries);
        }
        if (debugQueries > 0) {
            System.out.println("Debug queries: first " + debugQueries);
        }
        if (dumpTopKPath != null) {
            System.out.println("Dump topK: " + dumpTopKPath);
        }
        if (docsPath != null) {
            System.out.println("Docs (for recall): " + docsPath);
        } else {
            System.out.println("Recall: N/A (no --docs)");
        }
        System.out.println("K values: " + java.util.Arrays.toString(kValues));
        if (collaborative) {
            System.out.println("Collaborative HNSW: enabled (PR 15676; shared min-score across shards)");
            if (isolated) {
                System.out.println("Isolation: enabled (simulated coordinator push; per-shard bar)");
            }
        }
        if (progress) {
            System.out.println("Progress: enabled (every 1000 queries per K, to stderr)");
        }
        System.out.println();

        List<float[]> queries = List.of();
        List<float[]> docs = null;
        try {
            queries = VecReader.readAll(queriesPath, dim);
            if (docsPath != null) {
                docs = VecReader.readAllFromPath(docsPath, dim);
            }
            if (maxQueries > 0 && queries.size() > maxQueries) {
                queries = new ArrayList<>(queries.subList(0, maxQueries));
            }
        } catch (IOException e) {
            System.err.println("Failed to load vectors: " + e.getMessage());
            System.exit(1);
        }

        ExactNN exactNN = docs != null ? new ExactNN(docs, dim) : null;
        String datasetName = shardsDir.getFileName().toString();

        try (ShardKnnTester tester = new ShardKnnTester(
            shardPaths,
            IndexBuilder.VECTOR_FIELD,
            IndexBuilder.ID_FIELD,
            dim,
            effectiveSearchThreads
        );
             java.io.PrintWriter dumpWriter = dumpTopKPath != null
                 ? new java.io.PrintWriter(Files.newBufferedWriter(dumpTopKPath))
                 : null) {
            int nDoc = tester.getTotalDocs();
            int numShards = tester.getNumShards();
            int numQueries = queries.size();

            System.out.println("nDoc=" + nDoc + " numShards=" + numShards + " num_queries=" + numQueries);
            System.out.println();

            // recall = merged recall: (merged top-K from all shards) vs exact global top-K over full doc set
            System.out.println("topK\tlatency_ms\tmerged_recall\tlookups_saved\tnDoc\tnumShards\tnum_queries");
            List<String> summaryLines = new ArrayList<>();

            ExecutorService queryExecutor = queryThreads > 1 ? Executors.newFixedThreadPool(queryThreads) : null;
            final ShardKnnTester testerRef = tester;
            final List<float[]> queriesRef = queries;
            final ExactNN exactNNRef = exactNN;
            try {
                for (int K : kValues) {
                    if (progress) {
                        System.err.println("Running K=" + K + " (" + numQueries + " queries)...");
                    }
                    long totalLatencyMs;
                    double sumRecall;
                    int recallCount;
                    long totalLookupsSaved;
                    int lookupsSavedCount;
                    final AtomicInteger progressDone = progress ? new AtomicInteger(0) : null;
                    if (queryExecutor != null) {
                        final int kVal = K;
                        final boolean collaborativeRef = collaborative;
                        final boolean progressRef = progress;
                        final boolean isolatedRef = isolated;
                        final int debugQueriesRef = debugQueries;
                        final java.io.PrintWriter dumpWriterRef = dumpWriter;
                        int chunks = Math.min(queryThreads, numQueries);
                        int chunkSize = (numQueries + chunks - 1) / chunks;
                        List<Callable<ChunkResult>> tasks = new ArrayList<>(chunks);
                        for (int c = 0; c < chunks; c++) {
                            int start = c * chunkSize;
                            int end = Math.min(start + chunkSize, numQueries);
                            if (start >= end) continue;
                            final int chunkStart = start;
                            final int chunkEnd = end;
                            tasks.add(() -> runQueryChunk(
                                testerRef,
                                queriesRef,
                                exactNNRef,
                                kVal,
                                chunkStart,
                                chunkEnd,
                                collaborativeRef,
                                isolatedRef,
                                progressRef,
                                progressDone,
                                numQueries,
                                debugQueriesRef,
                                dumpWriterRef
                            ));
                        }
                        List<Future<ChunkResult>> futures = queryExecutor.invokeAll(tasks);
                        totalLatencyMs = 0;
                        sumRecall = 0;
                        recallCount = 0;
                        totalLookupsSaved = 0;
                        lookupsSavedCount = 0;
                        for (Future<ChunkResult> f : futures) {
                            ChunkResult cr = f.get();
                            totalLatencyMs += cr.totalLatencyMs;
                            sumRecall += cr.sumRecall;
                            recallCount += cr.recallCount;
                            totalLookupsSaved += cr.totalLookupsSaved;
                            lookupsSavedCount += cr.lookupsSavedCount;
                        }
                    } else {
                        totalLatencyMs = 0;
                        sumRecall = 0;
                        recallCount = 0;
                        totalLookupsSaved = 0;
                        lookupsSavedCount = 0;
                        for (int q = 0; q < queries.size(); q++) {
                            ShardKnnTester.Result res = tester.searchOne(queries.get(q), K, null, collaborative, isolated);
                            totalLatencyMs += res.elapsedMs;
                            if (res.lookupsSaved >= 0) {
                                totalLookupsSaved += res.lookupsSaved;
                                lookupsSavedCount++;
                            }
                            if (dumpWriter != null) {
                                dumpTopK(dumpWriter, K, q, res.topIds);
                            }
                            if (exactNN != null) {
                                int[] exact = exactNN.exactTopK(queries.get(q), K);
                                int overlap = overlapCount(res.topIds, exact);
                                sumRecall += (double) overlap / res.topIds.length;
                                recallCount++;
                                if (debugQueries > 0 && q < debugQueries) {
                                    System.err.println(
                                        "DEBUG K=" + K + " q=" + q + " overlap=" + overlap + "/" + res.topIds.length);
                                }
                            }
                            if (progress && (q + 1) % 1000 == 0) {
                                System.err.println("  K=" + K + " progress: " + (q + 1) + "/" + numQueries);
                            }
                        }
                    }
                    long avgLatencyMs = numQueries > 0 ? totalLatencyMs / numQueries : 0;
                    String recallStr = exactNN != null && recallCount > 0
                        ? String.format(Locale.ROOT, "%.4f", sumRecall / recallCount)
                        : "N/A";
                    String lookupsSavedStr = lookupsSavedCount > 0
                        ? String.valueOf(totalLookupsSaved / lookupsSavedCount)
                        : "N/A";
                    String line = K + "\t" + avgLatencyMs + "\t" + recallStr + "\t" + lookupsSavedStr + "\t" + nDoc + "\t" + numShards + "\t" + numQueries;
                    System.out.println(line);
                    summaryLines.add("SUMMARY\t" + datasetName + "\t" + line);
                }
            } finally {
                if (queryExecutor != null) {
                    queryExecutor.shutdown();
                }
            }

            System.out.println();
            for (String s : summaryLines) {
                System.out.println(s);
            }
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static final Object PROGRESS_LOCK = new Object();

    private static ChunkResult runQueryChunk(
            ShardKnnTester tester,
            List<float[]> queries,
            ExactNN exactNN,
            int K,
            int start,
            int end,
            boolean collaborative,
            boolean isolated,
            boolean progress,
            AtomicInteger progressDone,
            int numQueries,
            int debugQueries,
            java.io.PrintWriter dumpWriter) throws IOException {
        long totalLatencyMs = 0;
        double sumRecall = 0;
        int recallCount = 0;
        long totalLookupsSaved = 0;
        int lookupsSavedCount = 0;
        for (int q = start; q < end; q++) {
            ShardKnnTester.Result res = tester.searchOne(queries.get(q), K, null, collaborative, isolated);
            totalLatencyMs += res.elapsedMs;
            if (res.lookupsSaved >= 0) {
                totalLookupsSaved += res.lookupsSaved;
                lookupsSavedCount++;
            }
            if (dumpWriter != null) {
                dumpTopK(dumpWriter, K, q, res.topIds);
            }
            if (exactNN != null) {
                int[] exact = exactNN.exactTopK(queries.get(q), K);
                int overlap = overlapCount(res.topIds, exact);
                sumRecall += (double) overlap / res.topIds.length;
                recallCount++;
                if (debugQueries > 0 && q < debugQueries) {
                    synchronized (PROGRESS_LOCK) {
                        System.err.println(
                            "DEBUG K=" + K + " q=" + q + " overlap=" + overlap + "/" + res.topIds.length);
                    }
                }
            }
            if (progress && progressDone != null) {
                int d = progressDone.incrementAndGet();
                if (d % 1000 == 0) {
                    synchronized (PROGRESS_LOCK) {
                        System.err.println("  K=" + K + " progress: " + d + "/" + numQueries);
                    }
                }
            }
        }
        return new ChunkResult(totalLatencyMs, sumRecall, recallCount, totalLookupsSaved, lookupsSavedCount);
    }

    private static final class ChunkResult {
        final long totalLatencyMs;
        final double sumRecall;
        final int recallCount;
        final long totalLookupsSaved;
        final int lookupsSavedCount;

        ChunkResult(long totalLatencyMs, double sumRecall, int recallCount, long totalLookupsSaved, int lookupsSavedCount) {
            this.totalLatencyMs = totalLatencyMs;
            this.sumRecall = sumRecall;
            this.recallCount = recallCount;
            this.totalLookupsSaved = totalLookupsSaved;
            this.lookupsSavedCount = lookupsSavedCount;
        }
    }

    private static List<Path> discoverShards(Path parent) throws IOException {
        List<Path> out = new ArrayList<>();
        List<int[]> nameAndIndex = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(parent)) {
            for (Path p : stream) {
                if (!Files.isDirectory(p)) continue;
                String name = p.getFileName().toString();
                Matcher m = SHARD_DIR.matcher(name);
                if (m.matches()) {
                    nameAndIndex.add(new int[] { Integer.parseInt(m.group(1)) });
                }
            }
        }
        nameAndIndex.sort((a, b) -> Integer.compare(a[0], b[0]));
        for (int[] ni : nameAndIndex) {
            out.add(parent.resolve("shard-" + ni[0]));
        }
        return out;
    }

    private static void usage(String msg) {
        if (msg != null) {
            System.err.println(msg);
        }
        System.err.println("Usage: RunShardTest --shards <dir> --queries <path> --dim <n> [--docs <path>] [--k 10,100,1000,10000]");
        System.err.println("  --shards   Directory containing shard-0, shard-1, ...");
        System.err.println("  --queries  Path to queries.vec (raw float32, little-endian)");
        System.err.println("  --dim      Vector dimension");
        System.err.println("  --docs     Path to docs.vec or dataset dir (with docs.vec or docs-shard-*.vec) for merged recall");
        System.err.println("  merged_recall: (merged top-K from all shards) vs exact global top-K; the stat that matters for collaborative HNSW.");
        System.err.println("  lookups_saved: reported when Lucene exposes visit count (else N/A)");
        System.err.println("  --k        Comma-separated K values (default: 10,100,1000,10000)");
        System.err.println("  --search-threads  Threads for shard search (default: num shards, or numShards*queryThreads)");
        System.err.println("  --query-threads   Parallel queries (default: 1). Use 2-3 to use more cores.");
        System.err.println("  --max-queries     Limit to first N queries (for correctness/debug runs)");
        System.err.println("  --debug-queries   Print per-query overlap for first N queries (to stderr)");
        System.err.println("  --dump-topk       Write top-K IDs per query to a file (for diffing runs)");
        System.err.println("  --collaborative   Use collaborative HNSW (PR 15676): shared min-score across shards for pruning. Requires PR Lucene JAR.");
        System.err.println("  --isolated        Simulate coordinator push (per-shard bars). Requires --collaborative.");
        System.err.println("  --progress       Print progress every 1000 queries per K (to stderr). Does not affect recall or latency results.");
        System.exit(msg == null ? 0 : 1);
    }

    private static int overlapCount(int[] retrieved, int[] exactTopK) {
        java.util.HashSet<Integer> exactSet = new java.util.HashSet<>();
        for (int id : exactTopK) {
            exactSet.add(id);
        }
        int hit = 0;
        for (int id : retrieved) {
            if (exactSet.contains(id)) hit++;
        }
        return hit;
    }

    private static void dumpTopK(java.io.PrintWriter writer, int k, int q, int[] topIds) {
        StringBuilder sb = new StringBuilder();
        sb.append("K=").append(k).append("\tq=").append(q).append("\tids=");
        for (int i = 0; i < topIds.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(topIds[i]);
        }
        synchronized (writer) {
            writer.println(sb);
            writer.flush();
        }
    }
}
