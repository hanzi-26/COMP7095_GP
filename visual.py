# 17. 绘图对比
plt.figure(figsize=(20, 14))

# (1) 时间序列折线图（测试集）：实际数据与两种模型测试预测
plt.subplot(2, 2, 1)
sns.lineplot(data=test_reset.reset_index(), x="DATE", y="ACCIDENT_RATE", label="Actual (Test)", color="blue")
sns.lineplot(data=test_reset.reset_index(), x="DATE", y="RF_Prediction", label="RF Prediction (Test)", color="green", linestyle="--")
sns.lineplot(data=test_reset.reset_index(), x="DATE", y="GBT_Prediction", label="GBT Prediction (Test)", color="purple", linestyle="--")
plt.title("Test Data: Actual vs RF & GBT Predictions")
plt.xlabel("Date")
plt.ylabel("Accident Rate")
plt.legend()

# (2) 月份箱线图（测试集）：展示各月实际值与两种预测方法的分布
plt.subplot(2, 2, 2)
# 重置索引以便绘图
test_reset_plot = test_reset.reset_index()
# 将实际值和预测值合并到一个 DataFrame 中
melted_all = pd.melt(
    test_reset_plot,
    id_vars=["DATE", "MONTH"],
    value_vars=["ACCIDENT_RATE", "RF_Prediction", "GBT_Prediction"],
    var_name="Method",
    value_name="Accident_Rate"
)
# 将变量名称替换为更直观的标签
melted_all["Method"] = melted_all["Method"].replace({
    "ACCIDENT_RATE": "Actual",
    "RF_Prediction": "RF Prediction",
    "GBT_Prediction": "GBT Prediction"
})
# 绘制箱线图
sns.boxplot(
    data=melted_all,
    x="MONTH",
    y="Accident_Rate",
    hue="Method",
    palette={"Actual": "blue", "RF Prediction": "green", "GBT Prediction": "purple"}
)
plt.title("Monthly Accident Rate Distribution (Test Data)")
plt.xlabel("Month")
plt.ylabel("Accident Rate")
plt.legend(title="Dataset", loc="upper right")

# (3) 30-Day Rolling Average对比图
plt.subplot(2, 2, (3,4))
# 测试集滚动平均（基于 datetime 索引）
actual_rolling = test_reset['ACCIDENT_RATE'].rolling('30D', min_periods=1).mean()
rf_rolling_test = test_reset['RF_Prediction'].rolling('30D', min_periods=1).mean()
gbt_rolling_test = test_reset['GBT_Prediction'].rolling('30D', min_periods=1).mean()

# 2025未来数据滚动平均
rf_rolling_future = future_rf_pd['prediction'].rolling('30D', min_periods=1).mean()
gbt_rolling_future = future_gbt_pd['prediction'].rolling('30D', min_periods=1).mean()

plt.plot(test_reset.index, actual_rolling, label="Actual Rolling Avg (Test)", color="blue")
plt.plot(test_reset.index, rf_rolling_test, label="RF Test Prediction Rolling Avg", color="green", linestyle="--")
plt.plot(test_reset.index, gbt_rolling_test, label="GBT Test Prediction Rolling Avg", color="purple", linestyle="--")
# 用红色和橙色显示2025未来预测的滚动平均
plt.plot(future_rf_pd.index, rf_rolling_future, label="RF 2025 Prediction Rolling Avg", color="red", linestyle="--", linewidth=2)
plt.plot(future_gbt_pd.index, gbt_rolling_future, label="GBT 2025 Prediction Rolling Avg", color="orange", linestyle="--", linewidth=2)

plt.title("30-Day Rolling Average: Actual vs Predictions")
plt.xlabel("Date")
plt.ylabel("Rolling Average Accident Rate")
plt.legend()

plt.tight_layout()
plt.show()