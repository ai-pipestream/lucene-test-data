package lucenetestdata.indexbuilder;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.FSDirectory;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Minimal indexer: single docs.vec (little-endian float32) → N shard indices.
 * Use when you have a raw .vec file without meta.json.
 * For full datasets with meta.json, use BuildIndex instead.
 */
public final class SimpleIndexer {

    public static void main(String[] args) throws IOException {
        if (args.length < 4) {
            System.err.println("Usage: SimpleIndexer <vec_file> <dim> <num_shards> <output_dir>");
            System.exit(1);
        }

        Path vecFile = Paths.get(args[0]);
        int dim = Integer.parseInt(args[1]);
        int numShards = Integer.parseInt(args[2]);
        Path outputDir = Paths.get(args[3]);

        System.out.printf("Indexing %s into %d shards at %s (Dim: %d)%n", vecFile, numShards, outputDir, dim);

        long totalVectors = Files.size(vecFile) / (dim * 4L);
        long vectorsPerShard = (long) Math.ceil((double) totalVectors / numShards);

        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(vecFile.toFile())))) {
            for (int s = 0; s < numShards; s++) {
                Path shardPath = outputDir.resolve("shard-" + s);
                Files.createDirectories(shardPath);

                System.out.printf("Building Shard %d at %s...%n", s, shardPath);

                IndexWriterConfig iwc = new IndexWriterConfig();
                iwc.setRAMBufferSizeMB(256);

                try (IndexWriter writer = new IndexWriter(FSDirectory.open(shardPath), iwc)) {
                    for (int i = 0; i < vectorsPerShard; i++) {
                        float[] vector = new float[dim];
                        for (int j = 0; j < dim; j++) {
                            vector[j] = Float.intBitsToFloat(Integer.reverseBytes(dis.readInt()));
                        }

                        Document doc = new Document();
                        doc.add(new KnnFloatVectorField("vector", vector, VectorSimilarityFunction.COSINE));
                        long globalId = (long) s << 32 | i;
                        doc.add(new StoredField("id", globalId));

                        writer.addDocument(doc);

                        if (i > 0 && i % 10000 == 0) {
                            System.out.printf("  Shard %d: Indexed %d vectors...%n", s, i);
                        }

                        if (dis.available() == 0) break;
                    }
                    writer.commit();
                }
                if (dis.available() == 0) break;
            }
        }
        System.out.println("Indexing Complete.");
    }
}
