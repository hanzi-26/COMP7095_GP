from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("AccidentAnalysis").getOrCreate()

# 加载数据
df = spark.read.csv("new_data.csv", header=True, inferSchema=True)
df.printSchema()
df.describe().show()
from pyspark.sql.types import DoubleType, IntegerType, LongType, FloatType  # 添加类型导入
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.sql.functions import month, year, col
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.ml.regression import GBTRegressor


# 1. 时间特征提取（确保转换为整数类型）
df = df.withColumn("MONTH", month("DATE").cast(IntegerType())) \
       .withColumn("YEAR", year("DATE").cast(IntegerType()))

# 2. 处理空值（仅处理数值列）
# 自动识别数值列（包含Spark原生数值类型）
numeric_cols = [f.name for f in df.schema.fields
               if isinstance(f.dataType, (DoubleType, IntegerType, LongType, FloatType))]
print("识别到的数值列：", numeric_cols)

# 使用中位数填充代替0填充更安全
from pyspark.sql.functions import expr
for col_name in numeric_cols:
    median_value = df.selectExpr(f"percentile_approx({col_name}, 0.5)").first()[0]
    df = df.withColumn(col_name, expr(f"nvl({col_name}, {median_value})"))

# 3. 类别特征编码（添加处理未见类别策略）
indexer = StringIndexer(
    inputCols=["VEHICLE_CLASS_CODE", "CODE"],
    outputCols=["VEHICLE_CLASS_INDEX", "CODE_INDEX"],
    handleInvalid="keep"  # 处理未知类别
)
df = indexer.fit(df).transform(df)

# 4. 特征工程验证（带类型检查）
feature_cols = [col_name for col_name in df.columns
               if col_name not in ["DATE", "ACCIDENT_RATE", "VEHICLE_CLASS_CODE", "CODE"]
               and isinstance(df.schema[col_name].dataType, (DoubleType, IntegerType, LongType, FloatType))]

print("最终使用的特征列：", feature_cols)

# 5. 向量化处理（优化错误处理）
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="raw_features",
    handleInvalid="skip"  # 或 "keep" 根据需求
).setHandleInvalid("skip")

df = assembler.transform(df)

# 6. 标准化前检查（带方差日志）
from pyspark.sql.functions import variance, stddev

# 计算各列统计量
stats_df = df.agg(*(variance(col(c)).alias(f"{c}_var") for c in feature_cols),
                  *(stddev(col(c)).alias(f"{c}_std") for c in feature_cols))

# 打印有问题的列
variances = stats_df.select([col(c) for c in stats_df.columns if "_var" in c]).first()
zero_var_cols = [c.replace("_var", "") for c in variances.asDict() if variances[c] == 0]
print("零方差特征：", zero_var_cols)

# 7. 执行标准化（带异常捕获）
try:
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=True
    )
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
except Exception as e:
    print("标准化失败，错误信息：")
    print(str(e))
    print("建议检查：")
    print("- 确保输入向量没有NaN/Infinity值")
    print("- 确认所有特征方差大于0")

# 将数据采样转换为Pandas DataFrame（带内存保护）
try:
    sample_size = 5000  # 根据内存调整
    if df.count() > sample_size:
        sample_df = df.sample(fraction=sample_size/df.count(), seed=42).toPandas()
    else:
        sample_df = df.toPandas()
except Exception as e:
    print(f"数据采样失败：{str(e)}")
    sample_df = None

if sample_df is not None:
    # 1. 特征分布矩阵图（使用交互式可视化）
    !pip install plotly -q  # 如果未安装
    import plotly.express as px

    fig = px.scatter_matrix(
        sample_df[feature_cols[:5] + ["ACCIDENT_RATE"]],
        dimensions=feature_cols[:5] + ["ACCIDENT_RATE"],
        color="ACCIDENT_RATE",
        title="Feature Pair Plot"
    )
    fig.show()

    # 2. 时间序列分解分析
    plt.figure(figsize=(18, 8))

    # 年趋势
    plt.subplot(2, 2, 1)
    sns.lineplot(data=sample_df, x="YEAR", y="ACCIDENT_RATE", ci=95)
    plt.title("Yearly Trend")

    # 月趋势
    plt.subplot(2, 2, 2)
    sns.boxplot(data=sample_df, x="MONTH", y="ACCIDENT_RATE")
    plt.title("Monthly Distribution")

    # 滚动统计
    plt.subplot(2, 2, (3,4))
    sample_df.set_index(pd.to_datetime(sample_df['DATE'])).rolling('30D')['ACCIDENT_RATE'].mean().plot()
    plt.title("30-day Rolling Average")
    plt.tight_layout()

    # 3. 特征重要性分析（使用随机森林）
    from pyspark.ml.regression import RandomForestRegressor

    try:
        rf = RandomForestRegressor(
            featuresCol="features",
            labelCol="ACCIDENT_RATE",
            maxBins=100  # 处理类别型特征
        )
        model = rf.fit(df)

        # 可视化特征重要性
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.featureImportances.toArray()
        }).sort_values("importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(10), x="importance", y="feature")
        plt.title("Top 10 Important Features")

    except Exception as e:
        print(f"特征重要性分析失败：{str(e)}")
