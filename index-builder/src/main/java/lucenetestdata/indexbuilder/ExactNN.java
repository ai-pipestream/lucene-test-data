package lucenetestdata.indexbuilder;

import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Exact top-K by brute-force cosine (dot product on normalized vectors).
 * Used as ground truth for recall.
 */
public final class ExactNN {

    private final List<float[]> docs;
    private final int dim;

    public ExactNN(List<float[]> docs, int dim) {
        this.docs = docs;
        this.dim = dim;
    }

    /**
     * Return global doc ids of exact top-K for the given query vector (by cosine similarity).
     * Assumes vectors are normalized so dot product = cosine.
     */
    public int[] exactTopK(float[] queryVec, int K) {
        if (queryVec.length != dim) {
            throw new IllegalArgumentException("query dimension " + queryVec.length + " != " + dim);
        }
        K = Math.min(K, docs.size());
        if (K <= 0) return new int[0];
        // Min-heap of (score, docId); keep size K.
        PriorityQueue<float[]> heap = new PriorityQueue<>(K + 1, (a, b) -> Float.compare(a[0], b[0]));
        for (int i = 0; i < docs.size(); i++) {
            float score = dot(queryVec, docs.get(i));
            if (heap.size() < K) {
                heap.add(new float[] { score, i });
            } else if (score > heap.peek()[0]) {
                heap.poll();
                heap.add(new float[] { score, i });
            }
        }
        float[][] arr = heap.toArray(new float[0][]);
        Arrays.sort(arr, (a, b) -> Float.compare(b[0], a[0])); // desc by score
        int[] ids = new int[arr.length];
        for (int i = 0; i < arr.length; i++) {
            ids[i] = (int) arr[i][1];
        }
        return ids;
    }

    private static float dot(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
}
