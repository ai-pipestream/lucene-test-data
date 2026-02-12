package lucenetestdata.indexbuilder;

import java.lang.reflect.Constructor;
import java.util.concurrent.atomic.LongAccumulator;
import java.util.function.IntUnaryOperator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.knn.KnnCollectorManager;

/**
 * KNN query that uses Lucene's collaborative HNSW collector when available (PR 15676).
 * Shares a {@link LongAccumulator} across shards so later shards can prune using the
 * minimum score from earlier shards. Uses reflection so this compiles against stock
 * Lucene 10.3.2; when the PR JAR is on the classpath, collaborative search is used.
 */
public class CollaborativeKnnFloatVectorQuery extends KnnFloatVectorQuery {

    private final LongAccumulator minScoreAcc;
    private final IntUnaryOperator docIdMapper;
    private final int numShards;

    public CollaborativeKnnFloatVectorQuery(
            String field, float[] target, int k, LongAccumulator minScoreAcc) {
        this(field, target, k, minScoreAcc, null, 1);
    }

    public CollaborativeKnnFloatVectorQuery(
            String field,
            float[] target,
            int k,
            LongAccumulator minScoreAcc,
            IntUnaryOperator docIdMapper) {
        this(field, target, k, minScoreAcc, docIdMapper, 1);
    }

    public CollaborativeKnnFloatVectorQuery(
            String field,
            float[] target,
            int k,
            LongAccumulator minScoreAcc,
            IntUnaryOperator docIdMapper,
            int numShards) {
        super(field, target, k);
        this.minScoreAcc = minScoreAcc;
        this.docIdMapper = docIdMapper;
        this.numShards = numShards;
    }

    @Override
    protected KnnCollectorManager getKnnCollectorManager(int k, IndexSearcher searcher) {
        try {
            Class<?> clazz =
                    Class.forName("org.apache.lucene.search.knn.CollaborativeKnnCollectorManager");
            
            // Try 4-arg ctor (int, LongAccumulator, IntUnaryOperator, int)
            try {
                Constructor<?> ctor = clazz.getConstructor(
                        int.class, LongAccumulator.class, IntUnaryOperator.class, int.class);
                return (KnnCollectorManager) ctor.newInstance(k, minScoreAcc, 
                        docIdMapper != null ? docIdMapper : (IntUnaryOperator) (id -> id), 
                        numShards);
            } catch (NoSuchMethodException ignore) {
            }

            if (docIdMapper != null) {
                try {
                    Constructor<?> ctor = clazz.getConstructor(
                            int.class, LongAccumulator.class, IntUnaryOperator.class);
                    return (KnnCollectorManager) ctor.newInstance(k, minScoreAcc, docIdMapper);
                } catch (NoSuchMethodException ignore) {
                    // fall back to 2-arg ctor when mapper not supported
                }
            }
            Constructor<?> ctor = clazz.getConstructor(int.class, LongAccumulator.class);
            return (KnnCollectorManager) ctor.newInstance(k, minScoreAcc);
        } catch (ReflectiveOperationException e) {
            // PR JAR not on classpath; use default (no cross-shard pruning)
            return super.getKnnCollectorManager(k, searcher);
        }
    }
}
