package lucenetestdata.indexbuilder;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAccumulator;
import java.util.function.IntUnaryOperator;
import org.apache.lucene.search.TopDocs;

/**
 * Collaborative KNN query that tracks visited count via mergeLeafResults.
 */
public final class TrackingCollaborativeKnnFloatVectorQuery extends CollaborativeKnnFloatVectorQuery {
    private final AtomicLong totalVisitedCount = new AtomicLong();

    public TrackingCollaborativeKnnFloatVectorQuery(
            String field, float[] target, int k, LongAccumulator minScoreAcc) {
        super(field, target, k, minScoreAcc);
    }

    public TrackingCollaborativeKnnFloatVectorQuery(
            String field,
            float[] target,
            int k,
            LongAccumulator minScoreAcc,
            IntUnaryOperator docIdMapper) {
        super(field, target, k, minScoreAcc, docIdMapper);
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
