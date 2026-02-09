package lucenetestdata.indexbuilder;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Mimics a streaming push in a live system: one slot per shard, each shard updates its own slot
 * (like pushing an update). Others read all slots to get the current global top-K min score for pruning.
 * Each executor holds a reference to this object; no shared memory address—each update "copies" into
 * that shard's slot, like a push over the wire.
 */
public final class SharedPruningState {

    private final int numShards;
    private final AtomicReference<ShardSnapshot>[] slots;

    @SuppressWarnings("unchecked")
    public SharedPruningState(int numShards) {
        this.numShards = numShards;
        this.slots = new AtomicReference[numShards];
        for (int i = 0; i < numShards; i++) {
            slots[i] = new AtomicReference<>(ShardSnapshot.EMPTY);
        }
    }

    /**
     * Push this shard's latest result (like a streaming update). Each shard has its own slot;
     * this copies the result into that slot so others can read it.
     */
    public void update(int shardId, List<ShardKnnTester.ScoreDocWithGlobalId> results) {
        if (shardId < 0 || shardId >= numShards) return;
        ShardSnapshot snap = ShardSnapshot.from(results);
        slots[shardId].set(snap);
    }

    /**
     * Read current state from all shards, merge to global top-K, return the minimum score in that top-K.
     * Used by shards to decide if they can prune (exit early if they can't beat this score).
     */
    public float getGlobalMinTopKScore(int K) {
        List<ShardKnnTester.ScoreDocWithGlobalId> merged = new ArrayList<>();
        for (int i = 0; i < numShards; i++) {
            ShardSnapshot s = slots[i].get();
            if (s != null && s.scores != null) {
                for (int j = 0; j < s.len; j++) {
                    merged.add(new ShardKnnTester.ScoreDocWithGlobalId(s.docIds[j], s.scores[j]));
                }
            }
        }
        if (merged.isEmpty()) return Float.NEGATIVE_INFINITY;
        merged.sort(Comparator.comparingDouble((ShardKnnTester.ScoreDocWithGlobalId x) -> (double) x.score).reversed());
        int k = Math.min(K, merged.size());
        return merged.get(k - 1).score;
    }

    public int getNumShards() {
        return numShards;
    }

    /** Per-shard snapshot (own copy, like a pushed payload). */
    public static final class ShardSnapshot {
        static final ShardSnapshot EMPTY = new ShardSnapshot(null, null, 0);

        final int[] docIds;
        final float[] scores;
        final int len;

        ShardSnapshot(int[] docIds, float[] scores, int len) {
            this.docIds = docIds;
            this.scores = scores;
            this.len = len;
        }

        static ShardSnapshot from(List<ShardKnnTester.ScoreDocWithGlobalId> list) {
            if (list == null || list.isEmpty()) return EMPTY;
            int n = list.size();
            int[] docIds = new int[n];
            float[] scores = new float[n];
            for (int i = 0; i < n; i++) {
                ShardKnnTester.ScoreDocWithGlobalId x = list.get(i);
                docIds[i] = x.globalId;
                scores[i] = x.score;
            }
            return new ShardSnapshot(docIds, scores, n);
        }
    }
}
