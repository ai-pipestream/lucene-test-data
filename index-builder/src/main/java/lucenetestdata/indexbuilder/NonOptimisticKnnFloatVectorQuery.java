package lucenetestdata.indexbuilder;

import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.TopKnnCollectorManager;

/**
 * A KNN query that disables Lucene's "Optimistic" per-leaf K reduction.
 * This ensures that every shard searches for the full K, providing a fair
 * baseline for collaborative pruning which also searches for the full K.
 */
public class NonOptimisticKnnFloatVectorQuery extends KnnFloatVectorQuery {
    public NonOptimisticKnnFloatVectorQuery(String field, float[] target, int k) {
        super(field, target, k);
    }

    @Override
    protected KnnCollectorManager getKnnCollectorManager(int k, IndexSearcher searcher) {
        // By passing null for the searcher, TopKnnCollectorManager disables optimism
        // and uses the full k for every leaf.
        return new TopKnnCollectorManager(k, null);
    }
}