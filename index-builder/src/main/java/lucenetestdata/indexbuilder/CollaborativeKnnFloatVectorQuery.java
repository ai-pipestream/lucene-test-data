package lucenetestdata.indexbuilder;

import java.lang.reflect.Constructor;
import java.util.concurrent.atomic.LongAccumulator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.knn.KnnCollectorManager;

/**
 * KNN query that uses Lucene's collaborative HNSW collector when available (PR 15676).
 * Shares a {@link LongAccumulator} across shards so later shards can prune using the
 * minimum score from earlier shards. Uses reflection so this compiles against stock
 * Lucene 10.3.2; when the PR JAR is on the classpath, collaborative search is used.
 */
public final class CollaborativeKnnFloatVectorQuery extends KnnFloatVectorQuery {

    private final LongAccumulator minScoreAcc;

    public CollaborativeKnnFloatVectorQuery(
            String field, float[] target, int k, LongAccumulator minScoreAcc) {
        super(field, target, k);
        this.minScoreAcc = minScoreAcc;
    }

    @Override
    protected KnnCollectorManager getKnnCollectorManager(int k, IndexSearcher searcher) {
        try {
            Class<?> clazz =
                    Class.forName("org.apache.lucene.search.knn.CollaborativeKnnCollectorManager");
            Constructor<?> ctor = clazz.getConstructor(int.class, LongAccumulator.class);
            return (KnnCollectorManager) ctor.newInstance(k, minScoreAcc);
        } catch (ReflectiveOperationException e) {
            // PR JAR not on classpath; use default (no cross-shard pruning)
            return super.getKnnCollectorManager(k, searcher);
        }
    }
}
