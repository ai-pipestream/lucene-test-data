package lucenetestdata.indexbuilder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.LongAccumulator;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;

/**
 * Runs KNN search across multiple shard indices, merges top-K by score, and optionally computes recall vs exact NN.
 * When searchThreads > 1, shard searches run in parallel.
 */
public final class ShardKnnTester implements AutoCloseable {

    private final List<IndexSearcher> searchers;
    private final List<DirectoryReader> readers;
    private final String vectorField;
    private final String idField;
    private final int dim;
    private final ExecutorService searchExecutor;

    public ShardKnnTester(List<Path> shardDirs, String vectorField, String idField, int dim) throws IOException {
        this(shardDirs, vectorField, idField, dim, 1);
    }

    /**
     * @param searchThreads number of threads for parallel shard search (1 = sequential). Typically num shards or more.
     */
    public ShardKnnTester(List<Path> shardDirs, String vectorField, String idField, int dim, int searchThreads) throws IOException {
        this.vectorField = vectorField;
        this.idField = idField;
        this.dim = dim;
        this.readers = new ArrayList<>();
        this.searchers = new ArrayList<>();
        for (Path p : shardDirs) {
            DirectoryReader r = DirectoryReader.open(FSDirectory.open(p));
            readers.add(r);
            searchers.add(new IndexSearcher(r));
        }
        this.searchExecutor = searchThreads > 1
            ? Executors.newFixedThreadPool(Math.min(searchThreads, searchers.size()))
            : null;
    }

    public int getNumShards() {
        return searchers.size();
    }

    public int getTotalDocs() {
        int n = 0;
        for (DirectoryReader r : readers) {
            n += r.numDocs();
        }
        return n;
    }

    @Override
    public void close() throws IOException {
        if (searchExecutor != null) {
            searchExecutor.shutdown();
        }
        for (DirectoryReader r : readers) {
            r.close();
        }
    }

    /**
     * Search all shards for top-K, merge by score, return global doc ids and elapsed ms.
     * When sharedState is non-null, each shard pushes its result via sharedState.update(shardId, results)
     * so others can read live (mimics streaming push; each has its own slot).
     * When useCollaborative is true, uses a shared LongAccumulator and collaborative KNN query (PR 15676)
     * so later shards can prune; only effective when Lucene PR JAR is on classpath.
     */
    public Result searchOne(float[] queryVec, int K) throws IOException {
        return searchOne(queryVec, K, null, false);
    }

    public Result searchOne(float[] queryVec, int K, SharedPruningState sharedState) throws IOException {
        return searchOne(queryVec, K, sharedState, false);
    }

    public Result searchOne(float[] queryVec, int K, SharedPruningState sharedState, boolean useCollaborative) throws IOException {
        LongAccumulator minScoreAcc = useCollaborative ? new LongAccumulator(Long::max, Long.MIN_VALUE) : null;
        long start = System.nanoTime();
        List<ScoreDocWithGlobalId> all;
        if (searchExecutor != null) {
            List<Future<List<ScoreDocWithGlobalId>>> futures = new ArrayList<>(searchers.size());
            for (int s = 0; s < searchers.size(); s++) {
                final int shardIndex = s;
                final LongAccumulator acc = minScoreAcc;
                futures.add(searchExecutor.submit(() -> {
                    List<ScoreDocWithGlobalId> res = searchOneShard(shardIndex, queryVec, K, acc);
                    if (sharedState != null) {
                        sharedState.update(shardIndex, res);
                    }
                    return res;
                }));
            }
            all = new ArrayList<>();
            try {
                for (Future<List<ScoreDocWithGlobalId>> f : futures) {
                    all.addAll(f.get());
                }
            } catch (Exception e) {
                throw new IOException("Parallel shard search failed", e);
            }
        } else {
            all = new ArrayList<>();
            for (int s = 0; s < searchers.size(); s++) {
                List<ScoreDocWithGlobalId> res = searchOneShard(s, queryVec, K, minScoreAcc);
                if (sharedState != null) {
                    sharedState.update(s, res);
                }
                all.addAll(res);
            }
        }
        all.sort((a, b) -> Float.compare(b.score, a.score));
        int[] topIds = new int[Math.min(K, all.size())];
        for (int i = 0; i < topIds.length; i++) {
            topIds[i] = all.get(i).globalId;
        }
        long elapsedMs = (System.nanoTime() - start) / 1_000_000;
        return new Result(topIds, elapsedMs, -1L);
    }

    private List<ScoreDocWithGlobalId> searchOneShard(int shardIndex, float[] queryVec, int K) throws IOException {
        return searchOneShard(shardIndex, queryVec, K, null);
    }

    private List<ScoreDocWithGlobalId> searchOneShard(int shardIndex, float[] queryVec, int K, LongAccumulator minScoreAcc) throws IOException {
        KnnFloatVectorQuery query = minScoreAcc != null
            ? new CollaborativeKnnFloatVectorQuery(vectorField, queryVec, K, minScoreAcc)
            : new KnnFloatVectorQuery(vectorField, queryVec, K);
        TopDocs td = searchers.get(shardIndex).search(query, K);
        StoredFields stored = readers.get(shardIndex).storedFields();
        List<ScoreDocWithGlobalId> list = new ArrayList<>(td.scoreDocs.length);
        for (ScoreDoc sd : td.scoreDocs) {
            int globalId = stored.document(sd.doc).getField(idField).numericValue().intValue();
            list.add(new ScoreDocWithGlobalId(globalId, sd.score));
        }
        return list;
    }

    /** Recall = |retrieved ∩ exact| / K */
    public static float recall(int[] retrieved, int[] exactTopK) {
        if (retrieved.length == 0) return 0;
        Set<Integer> exactSet = new HashSet<>();
        for (int id : exactTopK) {
            exactSet.add(id);
        }
        int hit = 0;
        for (int id : retrieved) {
            if (exactSet.contains(id)) hit++;
        }
        return (float) hit / retrieved.length;
    }

    /**
     * Result of one sharded KNN search.
     * lookupsSaved: if the Lucene build exposes visit/distance count, set to (nDoc - visits) per query;
     * otherwise -1 (reported as N/A).
     */
    public static final class Result {
        public final int[] topIds;
        public final long elapsedMs;
        /** -1 if unknown (stock Lucene); else e.g. nDoc - visits for this query */
        public final long lookupsSaved;

        Result(int[] topIds, long elapsedMs) {
            this(topIds, elapsedMs, -1L);
        }

        Result(int[] topIds, long elapsedMs, long lookupsSaved) {
            this.topIds = topIds;
            this.elapsedMs = elapsedMs;
            this.lookupsSaved = lookupsSaved;
        }
    }

    /** Per-hit (globalId + score); package visibility for SharedPruningState. */
    static final class ScoreDocWithGlobalId {
        final int globalId;
        final float score;

        ScoreDocWithGlobalId(int globalId, float score) {
            this.globalId = globalId;
            this.score = score;
        }
    }
}
