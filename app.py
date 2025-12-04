import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import os

font_path = "NotoSansSC-VariableFont_wght.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    matplotlib.rcParams['font.family'] = font_name
    matplotlib.rcParams['axes.unicode_minus'] = False


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from datetime import datetime, timedelta

# 🎉 节假日列表（2025-12-04 起未来三年）
holiday_list = [
    # 2026 元旦
    "2026-01-01",
    # 2026 春节（示例）
    "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20", "2026-02-21", "2026-02-22",
    # 2026 清明节
    "2026-04-04", "2026-04-05", "2026-04-06",
    # 2026 劳动节
    "2026-05-01", "2026-05-02", "2026-05-03",
    # 2026 国庆节
    "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07",
    # 2027、2028 可补充
]
holiday_list = [pd.to_datetime(d) for d in holiday_list]

# 📅 场次生成函数
def generate_show_schedule(start_date, end_date, weekly_plan):
    """
    根据开始/结束日期和每周排期生成所有场次时间
    weekly_plan: dict like {"0": ["14:30", "19:30"], "1": [], ..., "6": ["19:30"]}
    """
    all_slots = []
    current = start_date
    while current <= end_date:
        weekday = str(current.weekday())  # 0=周一
        if weekday in weekly_plan:
            for time_str in weekly_plan[weekday]:
                dt_str = f"{current.strftime('%Y-%m-%d')} {time_str}"
                dt = pd.to_datetime(dt_str)
                all_slots.append(dt)
        current += timedelta(days=1)
    return sorted(all_slots)


st.set_page_config(layout="wide")
st.title("🎭 剧目营收预测系统")

def collect_cost_inputs():
    st.markdown("## 💰 成本参数设置（仅用于收益分析）")

    def input_group(title, items):
        st.markdown(f"#### {title}")
        values = {}
        for key, label, default in items:
            values[key] = st.number_input(f"{label}", value=default, step=100)
        return values

    # 一次性成本
    creation_costs = input_group("明细1 剧目创作", [
        ("1.1", "版权使用费", 0),
        ("1.2", "导演", 75000),
        ("1.3", "编剧/作词", 75000),
        ("1.4", "音乐创作及编曲", 75000),
        ("1.5", "音乐录制", 75000),
        ("1.6", "服装设计与制作", 50000),
        ("1.7", "灯光设计与落地", 30000),
        ("1.8", "编舞设计", 20000),
        ("1.9", "剧场及舞美设计", 50000),
        ("1.10", "音效设计", 30000),
        ("1.11", "声场声效设计", 50000),
        ("1.12", "多媒体设计", 0),
        ("1.13", "视觉设计", 20000),
        ("1.14", "应急预算", 50000),
    ])
    theater_costs = input_group("明细2 剧场相关", [
        ("2.1", "消防设计+施工", 300000),
        ("2.2", "硬装设计+施工+监控", 380000),
        ("2.3", "舞美软装道具+化妆间+吧台", 380000),
        ("2.4", "灯音麦等设备", 380000),
        ("2.5", "宽带网络", 10000),
        ("2.6", "物业费", 150000),
        ("2.7", "应急预算", 100000),
    ])
    rehearsal_costs = input_group("明细3 人员排练", [
        ("3.1", "大舞监（含行政运营）", 100000),
        ("3.2", "小舞监", 40000),
        ("3.3", "技术执行", 25000),
        ("3.4", "排练费", 75000),
        ("3.5", "排练场地", 30000),
        ("3.6", "卡米工资", 240000),
        ("3.7", "应急预算", 30000),
    ])
    promo_costs = input_group("明细4 宣发相关", [
        ("4.1", "剧目宣发及物料制作", 10000),
        ("4.2", "票务平台", 10000),
        ("4.3", "宣发营销", 10000),
        ("4.4", "应急预算", 10000),
    ])

    one_time_cost = sum(creation_costs.values()) + sum(theater_costs.values()) + sum(rehearsal_costs.values()) + sum(promo_costs.values())

    # 每场成本
    per_show_costs = input_group("🎭 每场演出成本", [
        ("演员", "演员", 6000),
        ("服化", "服化", 700),
        ("音控", "音控", 500),
        ("灯", "灯", 500),
        ("舞监", "舞监", 500),
        ("剧场经理", "剧场经理", 400),
        ("兼职", "兼职", 500),
        ("消耗型道具", "消耗型道具", 800),
        ("保洁", "保洁", 214.2),
    ])
    per_show_cost = sum(per_show_costs.values())

    # 每月管理费用
    st.markdown("#### 管理费用")
    monthly_admin = st.number_input("管理费用（固定）", value=120000)
    monthly_property = st.number_input("物业费用", value=0)
    monthly_cost = monthly_admin + monthly_property

    return one_time_cost, per_show_cost, monthly_cost




# 映射字典
type_map = {"话剧": 0, "音乐剧": 1}
resident_map = {"否": 0, "是": 1}
scale_map = {"小剧场": 0, "大中剧场": 1}
region_map = {
    "浦东新区": 0, "徐汇区": 1, "长宁区": 2, "普陀区": 3, "虹口区": 4, "杨浦区": 5,
    "黄浦区": 6, "静安区": 7, "宝山区": 8, "闵行区": 9, "嘉定区": 10, "松江区": 11,
    "金山区": 12, "青浦区": 13, "奉贤区": 14, "崇明区": 15
}
reverse_type_map = {v: k for k, v in type_map.items()}
reverse_resident_map = {v: k for k, v in resident_map.items()}
reverse_scale_map = {v: k for k, v in scale_map.items()}
reverse_region_map = {v: k for k, v in region_map.items()}

# 上传数据
# 上传数据
uploaded_file = st.file_uploader("📤 上传剧目场次数据文件（CSV）", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 预处理日期
    df["场次时间"] = pd.to_datetime(df["场次时间"])

    # 映射字段
    df["剧目类型"] = df["剧目类型"].map(type_map)
    df["是否常驻"] = df["是否常驻"].map({"否": 0, "是": 1})
    df["剧场规模"] = df["剧场规模"].map(scale_map)
    df["剧场区域"] = df["剧场区域"].map(region_map)

    # 特征列（排除剧目名称、场次时间、营业收入）
    exclude_cols = ["话剧名称", "场次时间", "营业收入"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # 特征与目标
    X_raw = df[feature_cols].copy()
    y_raw = df["营业收入"]

    # one-hot 编码（自动处理分类变量）
    X = pd.get_dummies(X_raw)

    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y_raw, test_size=0.2, random_state=42)

    # 模型选择
    model_options = ["Random Forest", "Ridge Regression", "XGBoost", "LightGBM", "MLP (多层感知机)"]
    model_name = st.selectbox("选择模型", model_options)

    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "Ridge Regression":
        model = Ridge()
    elif model_name == "XGBoost":
        model = XGBRegressor(n_estimators=100, random_state=42)
    elif model_name == "LightGBM":
        model = LGBMRegressor(n_estimators=100, random_state=42)
    elif model_name == "MLP (多层感知机)":
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

    # 模型训练
    model.fit(X_train, y_train)

    # 模型评分
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    st.success(f"模型 R² 分数：{score:.4f}")

    st.markdown("---")
    st.subheader("🎯 选择已有剧目进行预测")
    selected_name = st.selectbox("选择剧目", df["话剧名称"].unique())
    selected_rows = df[df["话剧名称"] == selected_name].copy()

    if not selected_rows.empty:
        # 特征处理
        X_selected = selected_rows[feature_cols].copy()
        X_selected = pd.get_dummies(X_selected)
        X_selected = X_selected.reindex(columns=X.columns, fill_value=0)

        # 预测
        y_pred = model.predict(X_selected)

        # 添加预测结果
        selected_rows["预测营收"] = y_pred
        selected_rows = selected_rows.sort_values("场次时间")

        # 图 1：单场次实际 vs 预测（条形图）
        st.subheader("📊 单场次实际营收 vs 预测营收")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        width = 0.4
        x = np.arange(len(selected_rows))

        ax1.bar(x - width/2, selected_rows["营业收入"], width=width, label="实际营收", color="#4CAF50")
        ax1.bar(x + width/2, selected_rows["预测营收"], width=width, label="预测营收", color="#2196F3")

        ax1.set_xticks(x)
        ax1.set_xticklabels(selected_rows["场次时间"].dt.strftime("%m-%d"), rotation=45)
        ax1.set_xlabel("场次时间")
        ax1.set_ylabel("营收（元）")
        ax1.set_title(f"{selected_name} 单场次营收对比")
        ax1.legend()
        ax1.grid(True, axis='y')
        fig1.tight_layout()
        st.pyplot(fig1)

        # 图 2：累计营收对比（折线图）
        st.subheader("📈 累计实际营收 vs 累计预测营收")
        fig2, ax2 = plt.subplots(figsize=(12, 5))

        selected_rows["累计实际营收"] = selected_rows["营业收入"].cumsum()
        selected_rows["累计预测营收"] = selected_rows["预测营收"].cumsum()

        ax2.plot(selected_rows["场次时间"], selected_rows["累计实际营收"], marker='o', label="累计实际营收", color="#4CAF50")
        ax2.plot(selected_rows["场次时间"], selected_rows["累计预测营收"], marker='s', label="累计预测营收", color="#2196F3")

        ax2.set_xlabel("场次时间")
        ax2.set_ylabel("累计营收（元）")
        ax2.set_title(f"{selected_name} 累计营收趋势对比")
        ax2.legend()
        ax2.grid(True)
        fig2.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")

    # 🆕 输入新剧信息进行预测
    with st.expander("🆕 输入新剧信息进行预测"):

        # 🎭 剧目参数设置（参与模型预测）
        st.markdown("### 🎭 剧目与场次参数设置")
    
        col1, col2, col3 = st.columns(3)
        with col1:
            show_type = st.selectbox("剧目类型", list(type_map.keys()))
        with col2:
            is_resident = st.selectbox("是否常驻", list(resident_map.keys()))
        with col3:
            scale = st.selectbox("剧场规模", list(scale_map.keys()))
    
        region = st.selectbox("剧场区域", list(region_map.keys()))
    
        st.markdown("### 📅 演出周期设置")
        today = pd.to_datetime("2025-12-04")
        max_date = today + pd.DateOffset(years=3)
    
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("开始日期", value=today.date(), min_value=today.date(), max_value=max_date.date())
        with col2:
            end_date = st.date_input("结束日期", value=(today + pd.Timedelta(days=30)).date(), min_value=today.date(), max_value=max_date.date())
    
        if end_date < start_date:
            st.warning("结束日期不能早于开始日期")
            st.stop()
    
        st.markdown("### 🗓 每周排期设置")
        weekday_map = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"}
        time_options = ["不演", "14:30", "19:30", "14:30 和 19:30"]
        weekly_plan = {}
    
        for i in range(7):
            choice = st.selectbox(f"{weekday_map[i]}", time_options, key=f"weekday_{i}")
            if choice == "14:30":
                weekly_plan[str(i)] = ["14:30"]
            elif choice == "19:30":
                weekly_plan[str(i)] = ["19:30"]
            elif choice == "14:30 和 19:30":
                weekly_plan[str(i)] = ["14:30", "19:30"]
            else:
                weekly_plan[str(i)] = []
    
        all_times = generate_show_schedule(pd.to_datetime(start_date), pd.to_datetime(end_date), weekly_plan)
        st.success(f"共生成 {len(all_times)} 场")
    
        st.markdown("### 🎬 剧目题材标签")
        all_tags = ["悬疑", "推理", "喜剧", "恐怖", "惊悚", "犯罪", "爱情", "历史", "传记",
                    "科幻", "奇幻", "玄幻", "灾难", "社会现实", "家庭伦理", "艺术文化", "战争", "职场", "其他"]
        selected_tags = st.multiselect("请选择题材标签（可多选）", options=all_tags)
        tag_values = {tag: (1 if tag in selected_tags else 0) for tag in all_tags}
    
        st.markdown("### 🎫 票价设置")
        col1, col2 = st.columns(2)
        with col1:
            max_price = st.number_input("最高票价", value=680)
        with col2:
            min_price = st.number_input("最低票价", value=80)
    
        # 💰 成本参数设置（不参与模型预测）
        st.markdown("### 💰 成本参数设置（仅用于收益分析）")
        col1, col2, col3 = st.columns(3)
        with col1:
            one_time_cost = st.number_input("一次性成本", value=50000)
        with col2:
            per_show_cost = st.number_input("每场演出成本", value=300)
        with col3:
            monthly_admin = st.number_input("每月管理成本", value=8000)
    
        # 🚀 开始预测
        if st.button("开始预测"):
            # 构建输入数据
            input_dict = {
                "剧目类型": type_map[show_type],
                "是否常驻": resident_map[is_resident],
                "剧场规模": scale_map[scale],
                "剧场区域": region_map[region]
            }
        
            schedule_df = pd.DataFrame({
                "场次时间": all_times,
                "星期几": [dt.weekday() for dt in all_times],
                "是否下午场": [1 if dt.hour == 14 else 0 for dt in all_times],
                "是否周末": [1 if dt.weekday() >= 5 else 0 for dt in all_times],
                "是否节假日": [1 if dt.normalize() in holiday_list else 0 for dt in all_times],
                "距开演首日的天数": [(dt - all_times[0]).days for dt in all_times],
                "最高价格": max_price,
                "最低价格": min_price,
                "周期": (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            })
        
            for k, v in input_dict.items():
                schedule_df[k] = v
            for tag, val in tag_values.items():
                schedule_df[tag] = val
        
            # one-hot 编码
            X_new = pd.get_dummies(schedule_df.drop(columns=["场次时间"]))
            X_new = X_new.reindex(columns=X.columns, fill_value=0)
        
            # 模型预测
            try:
                y_new = model.predict(X_new)
                schedule_df["预测营收"] = y_new
        
                # 成本与收益计算
                num_shows = len(schedule_df)
                period_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
                months = period_days / 30
                monthly_cost_total = monthly_cost * months
                total_cost = one_time_cost + per_show_cost * num_shows + monthly_cost_total
        
                per_show_fixed = one_time_cost / num_shows if num_shows > 0 else 0
                per_show_monthly = monthly_cost_total / num_shows if num_shows > 0 else 0
                schedule_df["每场成本"] = per_show_fixed + per_show_cost + per_show_monthly
                schedule_df["每场收益"] = schedule_df["预测营收"] - schedule_df["每场成本"]
                schedule_df["累计营收"] = schedule_df["预测营收"].cumsum()
                schedule_df["累计成本"] = schedule_df["每场成本"].cumsum()
                schedule_df["累计收益"] = schedule_df["每场收益"].cumsum()
        
                # 图1：预测营收
                st.subheader("📊 每场预测营收")
                fig1, ax1 = plt.subplots(figsize=(12, 5))
                ax1.bar(schedule_df["场次时间"], schedule_df["预测营收"], color="#2196F3")
                ax1.set_title("每场预测营收")
                ax1.set_xlabel("场次时间")
                ax1.set_ylabel("营收（元）")
                ax1.grid(True, axis='y')
                st.pyplot(fig1)
        
                # 图2：累计营收 vs 成本
                st.subheader("📈 累计营收 vs 累计成本")
                fig2, ax2 = plt.subplots(figsize=(12, 5))
                ax2.plot(schedule_df["场次时间"], schedule_df["累计营收"], label="累计营收", marker='o')
                ax2.plot(schedule_df["场次时间"], schedule_df["累计成本"], label="累计成本", marker='s')
                ax2.set_title("累计营收与成本对比")
                ax2.set_xlabel("场次时间")
                ax2.set_ylabel("金额（元）")
                ax2.legend()
                ax2.grid(True)
                st.pyplot(fig2)
        
                # 图3：每场收益 + 累计收益
                st.subheader("📉 每场收益与累计收益")
                fig3, ax3 = plt.subplots(figsize=(12, 5))
                ax3.bar(schedule_df["场次时间"], schedule_df["每场收益"], color="#4CAF50", label="每场收益")
                ax4 = ax3.twinx()
                ax4.plot(schedule_df["场次时间"], schedule_df["累计收益"], color="#FF5722", label="累计收益", marker='o')
                ax3.set_xlabel("场次时间")
                ax3.set_ylabel("每场收益", color="#4CAF50")
                ax4.set_ylabel("累计收益", color="#FF5722")
                fig3.legend(loc="upper left")
                fig3.tight_layout()
                st.pyplot(fig3)
        
                # 总结
                st.markdown(f"- 场次数：**{num_shows} 场**")
                st.markdown(f"- 预测总营收：**{schedule_df['预测营收'].sum():,.2f} 元**")
                st.markdown(f"- 总成本：**{total_cost:,.2f} 元**")
                st.markdown(f"- 预计利润：**{schedule_df['预测营收'].sum() - total_cost:,.2f} 元**")
        
                # 导出
                export_df = schedule_df[["场次时间", "预测营收", "每场成本", "每场收益", "累计营收", "累计成本", "累计收益"]]
                csv = export_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 下载预测结果 CSV",
                    data=csv,
                    file_name="预测结果.csv",
                    mime="text/csv"
                )
        
            except Exception as e:
                st.error(f"❌ 预测时出错：{e}")
                st.dataframe(X_new)
