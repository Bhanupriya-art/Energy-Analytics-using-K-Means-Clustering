from pyspark.sql import SparkSession
from pyspark.sql.types import NumericType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


spark = SparkSession.builder.appName("Energy_KMeans").getOrCreate()


df = spark.read.csv("global_energy_analytics.csv", header=True, inferSchema=True)

print("Initial Record Count:", df.count())

df.printSchema()
df = df.na.drop()


numeric_cols = [
    field.name
    for field in df.schema.fields
    if isinstance(field.dataType, NumericType)
]

print("Numeric Columns Used:", numeric_cols)


assembler = VectorAssembler(
    inputCols=numeric_cols,
    outputCol="features_unscaled"
)
assembled_df = assembler.transform(df)


scaler = StandardScaler(
    inputCol="features_unscaled",
    outputCol="features",
    withStd=True,
    withMean=True
)
scaler_model = scaler.fit(assembled_df)
scaled_df = scaler_model.transform(assembled_df)


kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=4, seed=42)
kmeans_model = kmeans.fit(scaled_df)

predictions = kmeans_model.transform(scaled_df)


evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster")
silhouette = evaluator.evaluate(predictions)

print("\n===== MODEL PERFORMANCE =====")
print(f"Silhouette Score: {silhouette:.4f}")


for i, center in enumerate(kmeans_model.clusterCenters()):
    print(f"\nCluster {i} center (first 10 values):")
    print(center[:10])


predictions.select(numeric_cols + ["cluster"]) \
           .toPandas().to_csv("energy_kmeans_results.csv", index=False)

print("\nSaved: energy_kmeans_results.csv")

spark.stop()
