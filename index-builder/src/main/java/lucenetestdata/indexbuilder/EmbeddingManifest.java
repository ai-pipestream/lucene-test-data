package lucenetestdata.indexbuilder;

import com.google.gson.annotations.SerializedName;
import java.util.Objects;

/**
 * Manifest (meta.json) for an embedding dataset produced by the Python script.
 * Describes docs.vec, queries.vec, dimension, and metadata.
 */
public final class EmbeddingManifest {

    private String source;
    @SerializedName("source_path")
    private String sourcePath;
    private String granularity;
    @SerializedName("model_name")
    private String modelName;
    private int dim;
    @SerializedName("num_docs")
    private int numDocs;
    @SerializedName("num_query_vectors")
    private int numQueryVectors;
    @SerializedName("output_docs_vec")
    private String outputDocsVec;
    @SerializedName("output_queries_vec")
    private String outputQueriesVec;
    @SerializedName("created_at")
    private String createdAt;
    @SerializedName("dataset_name")
    private String datasetName;
    @SerializedName("num_shards")
    private int numShards;
    @SerializedName("shard_sizes")
    private java.util.List<Integer> shardSizes;
    @SerializedName("shard_doc_offsets")
    private java.util.List<Integer> shardDocOffsets;

    public String getSource() { return source; }
    public void setSource(String source) { this.source = source; }

    public String getSourcePath() { return sourcePath; }
    public void setSourcePath(String sourcePath) { this.sourcePath = sourcePath; }

    public String getGranularity() { return granularity; }
    public void setGranularity(String granularity) { this.granularity = granularity; }

    public String getModelName() { return modelName; }
    public void setModelName(String modelName) { this.modelName = modelName; }

    public int getDim() { return dim; }
    public void setDim(int dim) { this.dim = dim; }

    public int getNumDocs() { return numDocs; }
    public void setNumDocs(int numDocs) { this.numDocs = numDocs; }

    public int getNumQueryVectors() { return numQueryVectors; }
    public void setNumQueryVectors(int numQueryVectors) { this.numQueryVectors = numQueryVectors; }

    public String getOutputDocsVec() { return outputDocsVec; }
    public void setOutputDocsVec(String outputDocsVec) { this.outputDocsVec = outputDocsVec; }

    public String getOutputQueriesVec() { return outputQueriesVec; }
    public void setOutputQueriesVec(String outputQueriesVec) { this.outputQueriesVec = outputQueriesVec; }

    public String getCreatedAt() { return createdAt; }
    public void setCreatedAt(String createdAt) { this.createdAt = createdAt; }

    public String getDatasetName() { return datasetName; }
    public void setDatasetName(String datasetName) { this.datasetName = datasetName; }

    public int getNumShards() { return numShards; }
    public java.util.List<Integer> getShardSizes() { return shardSizes; }
    public java.util.List<Integer> getShardDocOffsets() { return shardDocOffsets; }
    public boolean isSharded() { return numShards > 1; }
}
