hdfs dfs -mkdir -p /user/hadoop/data
hdfs dfs -put /home/hadoop/hadoop-data/new_data.csv /user/hadoop/data/

CREATE TABLE new_data (
    PDS INT,
    PSG INT,
    DRI INT,
    T_TFC_ACC INT,
    DATE_STR STRING,
    SOA_FATAL INT,
    SOA_SERIOUS INT,
    SOA_SLIGHT INT,
    VCT_VEHICLE INT,
    VCT_PDS INT,
    VCT_OBJ INT,
    VCT_NONCOL INT,
    NO_ISSUE_WARN_8_14PT INT,
    NO_ISSUE_SINCE_1984 INT,
    NO_SUMMON_15PT_MORE INT,
    NO_SUMMON_SINCE_1984 INT,
    NO_DRI_INCUR_PT_5YR INT,
    NO_VALID_LIC_TYPE1 INT,
    NO_VALID_LIC_TYPE2 INT,
    NO_VALID_LIC_TYPE3 INT,
    NO_VALID_LIC_TYPE4 INT,
    NO_NOVER_3YR_EXP_LIC_TYPE1 INT,
    NO_NOVER_3YR_EXP_LIC_TYPE2 INT,
    NO_NOVER_3YR_EXP_LIC_TYPE3 INT,
    NO_OVER_3YR_EXP_LIC_TYPE1 INT,
    NO_OVER_3YR_EXP_LIC_TYPE2 INT,
    NO_OVER_3YR_EXP_LIC_TYPE3 INT,
    VEHICLE_CLASS_CODE INT,
    CODE INT,
    NO_VALID_LEARNER INT,
    NO_VALID_FULL_LIC INT,
    NO_EXP_LEARNER INT,
    NO_EXP_NOVER_3YR INT,
    NO_EXP_OVER_3YR INT,
    ACCIDENT_RATE DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE;

LOAD DATA INPATH '/user/hadoop/data/new_data.csv' INTO TABLE temp_new_data;

SELECT
    AVG(PDS) AS avg_pds,
    SUM(PSG) AS total_psg,
    MAX(DRI) AS max_dri,
    MIN(T_TFC_ACC) AS min_t_tfc_acc,
    AVG(SOA_FATAL) AS avg_soa_fatal,
    SUM(SOA_SERIOUS) AS total_soa_serious,
    SUM(SOA_SLIGHT) AS total_soa_slight
FROM temp_new_data;

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
val assembler = new VectorAssembler()
  .setInputCols(Array("PDS", "PSG", "DRI", "T_TFC_ACC", "SOA_FATAL", "SOA_SERIOUS", "SOA_SLIGHT"))
  .setOutputCol("features")
val featureData = assembler.transform(data)
val kmeans = new KMeans()
  .setK(3)
  .setSeed(1L)
val model = kmeans.fit(featureData)
val predictions = model.transform(featureData)
predictions.show()

import org.apache.spark.ml.regression.LinearRegression
val assembler = new VectorAssembler()
  .setInputCols(Array("PDS", "PSG", "DRI", "T_TFC_ACC", "SOA_FATAL", "SOA_SERIOUS", "SOA_SLIGHT"))
  .setOutputCol("features")
val featureData = assembler.transform(data)
val Array(trainingData, testData) = featureData.randomSplit(Array(0.5, 0.5), seed = 1234L)
val lr = new LinearRegression()
  .setLabelCol("ACCIDENT_RATE")
  .setFeaturesCol("features")
val lrModel = lr.fit(trainingData)
val predictions = lrModel.transform(testData)
predictions.select("prediction", "ACCIDENT_RATE").show()

import org.apache.spark.ml.evaluation.RegressionEvaluator
val evaluator = new RegressionEvaluator()
  .setLabelCol("ACCIDENT_RATE")
  .setPredictionCol("prediction")
  .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
val r2Evaluator = new RegressionEvaluator()
  .setLabelCol("ACCIDENT_RATE")
  .setPredictionCol("prediction")
  .setMetricName("r2")
val r2 = r2Evaluator.evaluate(predictions)
println(s"R-squared (R²) on test data = $r2")

import matplotlib.pyplot as plt
plt.scatter(results["ACCIDENT_RATE"], results["prediction"])
plt.xlabel("Actual Accident Rate")
plt.ylabel("Predicted Accident Rate")
plt.title("Regression Results")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.residplot(x="ACCIDENT_RATE", y="prediction", data=results)
plt.xlabel("Actual Accident Rate")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()