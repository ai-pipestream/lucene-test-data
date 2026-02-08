package lucenetestdata.indexbuilder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Reads .vec files (luceneutil format: raw little-endian float32, no header).
 */
public final class VecReader {

    private VecReader() {}

    /**
     * Read all vectors from a .vec file.
     *
     * @param path path to the .vec file
     * @param dim  vector dimension (must match file size: file size must be divisible by dim * 4)
     * @return list of vectors (each of length dim)
     */
    public static List<float[]> readAll(Path path, int dim) throws IOException {
        long size = java.nio.file.Files.size(path);
        int vecSizeBytes = dim * Float.BYTES;
        if (size % vecSizeBytes != 0) {
            throw new IllegalArgumentException(
                "Vector file size " + size + " is not a multiple of vector size " + vecSizeBytes + " (dim=" + dim + ")");
        }
        int numVectors = (int) (size / vecSizeBytes);
        List<float[]> vectors = new ArrayList<>(numVectors);
        ByteBuffer buf = ByteBuffer.allocate(vecSizeBytes).order(ByteOrder.LITTLE_ENDIAN);
        try (FileChannel ch = FileChannel.open(path)) {
            for (int i = 0; i < numVectors; i++) {
                buf.clear();
                int read = ch.read(buf);
                if (read != vecSizeBytes) {
                    throw new IOException("Short read at vector " + i + ": expected " + vecSizeBytes + ", got " + read);
                }
                buf.flip();
                float[] vec = new float[dim];
                for (int j = 0; j < dim; j++) {
                    vec[j] = buf.getFloat();
                }
                vectors.add(vec);
            }
        }
        return vectors;
    }
}
