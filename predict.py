from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, LongType, FloatType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, Imputer
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.sql.functions import month, year, col, to_date, lit, make_date, avg
from functools import reduce
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. 初始化Spark会话并加载数据
spark = SparkSession.builder.appName("AccidentAnalysis").getOrCreate()
df = spark.read.csv("/content/new_data.csv", header=True, inferSchema=True)

# 2. 日期格式转换，并提取年份、月份信息
df = df.withColumn("DATE", to_date(col("DATE"), "yyyy/M/d"))
df = df.withColumn("YEAR", year(col("DATE")).cast(IntegerType()))
df = df.withColumn("MONTH", month(col("DATE")).cast(IntegerType()))

# 3. 只用历史数据进行训练（不包括2025年及之后的数据）
historical_df = df.filter(col("YEAR") < 2025)

# 4. 提取数值型特征（假设目标为 "ACCIDENT_RATE"）
numeric_cols = [f.name for f in historical_df.schema.fields
                if isinstance(f.dataType, (DoubleType, IntegerType, LongType, FloatType))]

# 5. 缺失值填充
imputer = Imputer(inputCols=numeric_cols, outputCols=numeric_cols, strategy="median")

# 6. 如果存在类别特征（如 VEHICLE_CLASS_CODE、CODE）则进行编码
categorical_columns = []
if "VEHICLE_CLASS_CODE" in historical_df.columns and "CODE" in historical_df.columns:
    categorical_columns = ["VEHICLE_CLASS_CODE", "CODE"]
    indexer = StringIndexer(
        inputCols=categorical_columns,
        outputCols=[col + "_INDEX" for col in categorical_columns],
        handleInvalid="keep"
    )
    feature_cols = numeric_cols + [col + "_INDEX" for col in categorical_columns]
else:
    feature_cols = numeric_cols

# 7. 向量化与标准化
assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features", handleInvalid="skip")
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)

# 8. 定义两种回归模型：随机森林（RF）与梯度提升树（GBT）
rf = RandomForestRegressor(featuresCol="features", labelCol="ACCIDENT_RATE", numTrees=50)
gbt = GBTRegressor(featuresCol="features", labelCol="ACCIDENT_RATE", maxIter=50)

# 9. 构建两个Pipeline
# Pipeline for Random Forest (RF)
stages_rf = [imputer]
if categorical_columns:
    stages_rf.append(indexer)
stages_rf += [assembler, scaler, rf]
pipeline_rf = Pipeline(stages=stages_rf)

# Pipeline for Gradient Boosted Trees (GBT)
stages_gbt = [imputer]
if categorical_columns:
    stages_gbt.append(indexer)
stages_gbt += [assembler, scaler, gbt]
pipeline_gbt = Pipeline(stages=stages_gbt)

# 10. 划分历史数据为70%训练与30%测试（随机划分）
train_df, test_df = historical_df.randomSplit([0.7, 0.3], seed=42)

# 11. 训练两种模型
model_rf = pipeline_rf.fit(train_df)
model_gbt = pipeline_gbt.fit(train_df)

# 12. 分别在测试集上进行预测
predictions_rf = model_rf.transform(test_df)
predictions_gbt = model_gbt.transform(test_df)

# 13. 构造2025年未来数据样本
# 利用历史数据按月份计算各月均值（这里只对数值型特征求均值）
monthly_avg_expr = [avg(c).alias(c) for c in numeric_cols if c not in ["MONTH", "YEAR"]]
monthly_avg = historical_df.groupBy("MONTH").agg(*monthly_avg_expr)
future_data_list = []
for m in range(1, 13):
    month_avg = monthly_avg.filter(col("MONTH") == m)
    future_row = month_avg.withColumn("YEAR", lit(2025)) \
                          .withColumn("MONTH", lit(m)) \
                          .withColumn("DATE", make_date(lit(2025), lit(m), lit(1)))
    future_data_list.append(future_row)
future_2025_df = reduce(lambda a, b: a.union(b), future_data_list)

# 14. 对2025未来数据分别用两种模型预测
future_pred_rf = model_rf.transform(future_2025_df)
future_pred_gbt = model_gbt.transform(future_2025_df)

# 15. 转换预测结果为Pandas DataFrame（测试集和未来数据分别转换）
# 测试集结果
test_rf_pd = predictions_rf.select("DATE", "YEAR", "MONTH", "ACCIDENT_RATE", "prediction")\
                           .orderBy("DATE").toPandas()
test_rf_pd['DATE'] = pd.to_datetime(test_rf_pd['DATE'])
test_rf_pd = test_rf_pd.sort_values("DATE").reset_index(drop=True)

test_gbt_pd = predictions_gbt.select("DATE", "YEAR", "MONTH", "ACCIDENT_RATE", "prediction")\
                             .orderBy("DATE").toPandas()
test_gbt_pd['DATE'] = pd.to_datetime(test_gbt_pd['DATE'])
test_gbt_pd = test_gbt_pd.sort_values("DATE").reset_index(drop=True)

# 2025预测结果
future_rf_pd = future_pred_rf.select("DATE", "YEAR", "MONTH", "ACCIDENT_RATE", "prediction")\
                              .orderBy("DATE").toPandas()
future_rf_pd['DATE'] = pd.to_datetime(future_rf_pd['DATE'])
future_rf_pd = future_rf_pd.sort_values("DATE").reset_index(drop=True)

future_gbt_pd = future_pred_gbt.select("DATE", "YEAR", "MONTH", "ACCIDENT_RATE", "prediction")\
                                .orderBy("DATE").toPandas()
future_gbt_pd['DATE'] = pd.to_datetime(future_gbt_pd['DATE'])
future_gbt_pd = future_gbt_pd.sort_values("DATE").reset_index(drop=True)

# 16. 对测试集数据添加预测结果（注意这里通过 .values 避免重复索引问题）
test_reset = test_rf_pd.copy()
test_reset["RF_Prediction"] = test_rf_pd["prediction"].values
test_reset["GBT_Prediction"] = test_gbt_pd["prediction"].values

# 为了计算基于日期的 rolling，需要把 DATE 设置为索引
test_reset = test_reset.set_index("DATE")
future_rf_pd = future_rf_pd.set_index("DATE")
future_gbt_pd = future_gbt_pd.set_index("DATE")