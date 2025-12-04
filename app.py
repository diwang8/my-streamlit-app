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
    st.markdown("## 💰 成本参数设置")

    st.markdown("### 一次性投入成本")
    col1, col2, col3 = st.columns(3)
    with col1:
        cost_director = st.number_input("导演", value=75000)
        cost_writer = st.number_input("编剧/作词", value=75000)
        cost_music = st.number_input("音乐创作及编曲", value=75000)
        cost_recording = st.number_input("音乐录制", value=75000)
        cost_costume = st.number_input("服装设计与制作", value=50000)
    with col2:
        cost_light = st.number_input("灯光设计与落地", value=30000)
        cost_choreography = st.number_input("编舞设计", value=20000)
        cost_stage = st.number_input("剧场及舞美设计", value=50000)
        cost_audio = st.number_input("音效设计", value=30000)
        cost_acoustics = st.number_input("声场声效设计", value=50000)
    with col3:
        cost_visual = st.number_input("视觉设计", value=20000)
        cost_emergency = st.number_input("应急预算（创作）", value=50000)
        cost_fire = st.number_input("消防设计+施工", value=300000)
        cost_hard = st.number_input("硬装设计+施工+监控", value=380000)
        cost_soft = st.number_input("舞美软装道具+化妆间+吧台", value=380000)

    one_time_cost = sum([
        cost_director, cost_writer, cost_music, cost_recording, cost_costume,
        cost_light, cost_choreography, cost_stage, cost_audio, cost_acoustics,
        cost_visual, cost_emergency, cost_fire, cost_hard, cost_soft
    ])

    st.markdown("### 持续性投入成本（单位：元/场）")
    col4, col5, col6 = st.columns(3)
    with col4:
        cost_actor = st.number_input("演员", value=6000)
        cost_makeup = st.number_input("服化", value=700)
        cost_audio_op = st.number_input("音控", value=500)
    with col5:
        cost_light_op = st.number_input("灯", value=500)
        cost_stage_mgr = st.number_input("舞监", value=500)
        cost_manager = st.number_input("剧场经理", value=400)
    with col6:
        cost_parttime = st.number_input("兼职", value=500)
        cost_props = st.number_input("消耗型道具", value=800)
        cost_cleaning = st.number_input("保洁", value=214.29)

    per_show_cost = sum([
        cost_actor, cost_makeup, cost_audio_op, cost_light_op,
        cost_stage_mgr, cost_manager, cost_parttime, cost_props, cost_cleaning
    ])

    st.markdown("### 管理费用")
    monthly_admin = st.number_input("管理费用（元/月）", value=120000)

    return one_time_cost, per_show_cost, monthly_admin



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
    
                # 📊 可视化
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(schedule_df["场次时间"], schedule_df["预测营收"], marker='o', color="#2196F3")
                ax.set_title("新剧每场次预测营收")
                ax.set_xlabel("场次时间")
                ax.set_ylabel("预测营收")
                st.pyplot(fig)
    
                # 💵 收益分析
                st.subheader("💵 成本与收益分析")
                num_shows = len(schedule_df)
                period = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
                admin_cost = monthly_admin * (period / 30)
                recurring_cost = per_show_cost * num_shows
                total_cost = one_time_cost + recurring_cost + admin_cost
                total_revenue = schedule_df["预测营收"].sum()
    
                st.markdown(f"- 场次数：**{num_shows} 场**")
                st.markdown(f"- 预测总营收：**{total_revenue:,.2f} 元**")
                st.markdown(f"- 总成本：**{total_cost:,.2f} 元**")
                st.markdown(f"- 预计利润：**{total_revenue - total_cost:,.2f} 元**")
    
                # 💾 导出
                export_df = schedule_df[["场次时间", "预测营收"]].copy()
                export_df["累计预测营收"] = export_df["预测营收"].cumsum()
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

