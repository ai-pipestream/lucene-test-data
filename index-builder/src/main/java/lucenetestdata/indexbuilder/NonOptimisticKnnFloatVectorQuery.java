package lucenetestdata.indexbuilder;

import java.io.IOException;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;

/**
 * KNN query that disables Lucene's optimistic per-leaf K collection.
 * This matches the "non-optimistic" path used in Lucene's collaborative HNSW tests.
 */
public class NonOptimisticKnnFloatVectorQuery extends KnnFloatVectorQuery {

    public NonOptimisticKnnFloatVectorQuery(String field, float[] target, int k) {
        super(field, target, k);
    }

    @Override
    protected KnnCollectorManager getKnnCollectorManager(int k, IndexSearcher searcher) {
        return new KnnCollectorManager() {
            @Override
            public KnnCollector newCollector(
                    int visitedLimit, KnnSearchStrategy searchStrategy, LeafReaderContext context)
                    throws IOException {
                return new TopKnnCollector(k, visitedLimit, searchStrategy);
            }

            @Override
            public boolean isOptimistic() {
                return false;
            }
        };
    }
}
