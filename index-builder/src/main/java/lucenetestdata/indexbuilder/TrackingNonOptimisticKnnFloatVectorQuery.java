package lucenetestdata.indexbuilder;

import java.util.concurrent.atomic.AtomicLong;
import org.apache.lucene.search.TopDocs;

/**
 * Non-optimistic KNN query that tracks visited count via mergeLeafResults.
 */
public final class TrackingNonOptimisticKnnFloatVectorQuery extends NonOptimisticKnnFloatVectorQuery {
    private final AtomicLong totalVisitedCount = new AtomicLong();

    public TrackingNonOptimisticKnnFloatVectorQuery(String field, float[] target, int k) {
        super(field, target, k);
    }

    @Override
    protected TopDocs mergeLeafResults(TopDocs[] perLeafResults) {
        long visited = 0;
        for (TopDocs td : perLeafResults) {
            if (td != null && td.totalHits != null) {
                visited += td.totalHits.value();
            }
        }
        totalVisitedCount.set(visited);
        return super.mergeLeafResults(perLeafResults);
    }

    long getTotalVisitedCount() {
        return totalVisitedCount.get();
    }
}
