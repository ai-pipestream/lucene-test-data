package lucenetestdata.indexbuilder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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

    private static final Pattern DOCS_SHARD = Pattern.compile("docs-shard-(\\d+)\\.vec");

    private static int shardIndex(String fileName) {
        Matcher m = DOCS_SHARD.matcher(fileName);
        return m.matches() ? Integer.parseInt(m.group(1)) : -1;
    }

    /**
     * Read all document vectors from a path: either a single .vec file or a dataset directory.
     * If path is a directory: reads docs.vec if present; otherwise reads docs-shard-0.vec, docs-shard-1.vec, ...
     * in order and concatenates (for recall vs exact NN).
     */
    public static List<float[]> readAllFromPath(Path path, int dim) throws IOException {
        if (Files.isRegularFile(path)) {
            return readAll(path, dim);
        }
        if (!Files.isDirectory(path)) {
            throw new IllegalArgumentException("Not a file or directory: " + path);
        }
        Path singleDocs = path.resolve("docs.vec");
        if (Files.isRegularFile(singleDocs)) {
            return readAll(singleDocs, dim);
        }
        List<Path> shardPaths = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(path)) {
            for (Path p : stream) {
                if (!Files.isRegularFile(p)) continue;
                String name = p.getFileName().toString();
                Matcher m = DOCS_SHARD.matcher(name);
                if (m.matches()) {
                    shardPaths.add(p);
                }
            }
        }
        if (shardPaths.isEmpty()) {
            throw new IllegalArgumentException("No docs.vec or docs-shard-*.vec in " + path);
        }
        shardPaths.sort((a, b) -> {
            int i = shardIndex(a.getFileName().toString());
            int j = shardIndex(b.getFileName().toString());
            return Integer.compare(i, j);
        });
        List<float[]> all = new ArrayList<>();
        for (Path p : shardPaths) {
            all.addAll(readAll(p, dim));
        }
        return all;
    }
}
