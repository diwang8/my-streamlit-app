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
    # 元旦：1月1日（周四）至3日（周六）
    "2026-01-01", "2026-01-02", "2026-01-03",

    # 春节：2月15日（周日）至23日（周一）
    "2026-02-15", "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19",
    "2026-02-20", "2026-02-21", "2026-02-22", "2026-02-23",

    # 清明节：4月4日（周六）至6日（周一）
    "2026-04-04", "2026-04-05", "2026-04-06",

    # 劳动节：5月1日（周五）至3日（周日）
    "2026-05-01", "2026-05-02", "2026-05-03",

    # 端午节：6月22日（周三）至24日（周五）
    "2026-06-22", "2026-06-23", "2026-06-24",

    # 中秋节：9月29日（周五）至10月1日（周日）
    "2026-09-29", "2026-09-30", "2026-10-01",

    # 国庆节：10月1日（周六）至7日（周五）
    "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04",
    "2026-10-05", "2026-10-06", "2026-10-07",
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

def generate_schedule_df(
    start_date, end_date, weekly_plan, input_dict, tag_values,
    max_price, min_price, holiday_list
):
    all_times = generate_show_schedule(pd.to_datetime(start_date), pd.to_datetime(end_date), weekly_plan)
    if not all_times:
        return None

    df = pd.DataFrame({
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
        df[k] = [v] * len(df)
    for tag, val in tag_values.items():
        df[tag] = val

    return df


def suggest_parameter_adjustments(
    model, X_columns, one_time_cost, per_show_cost, monthly_admin,
    investor_share_payback, investor_share_profit, venue_share, tax_rate, channel_share,
    start_date, end_date, target_days,
    input_dict, tag_values, selected_optimizable, weekly_plan, holiday_list,
    max_price, min_price
):
    suggestions = {}

    def simulate(input_dict, tag_values, start_date, end_date, weekly_plan, max_price, min_price):
        try:
            df = generate_schedule_df(
                start_date, end_date, weekly_plan,
                input_dict, tag_values,
                max_price, min_price, holiday_list
            )
            if df is None:
                return None

            X_new = pd.get_dummies(df.drop(columns=["场次时间"]))
            X_new = X_new.reindex(columns=X_columns, fill_value=0)
            y_pred = model.predict(X_new)
            df["预测营收"] = y_pred * (1 - venue_share - tax_rate - channel_share)

            num_shows = len(df)
            period_days = (df["场次时间"].max() - df["场次时间"].min()).days + 1
            admin_cost = monthly_admin * (period_days / 30)
            admin_per_show = admin_cost / num_shows
            df["每场收益"] = df["预测营收"] - (per_show_cost + admin_per_show)

            cumulative_profit = 0
            investor_share_list = []
            for profit in df["每场收益"]:
                cumulative_profit += profit
                investor_ratio = investor_share_payback if cumulative_profit < one_time_cost else investor_share_profit
                investor_share_list.append(profit * investor_ratio)

            df["投资者收益"] = investor_share_list
            df["累计投资者收益"] = df["投资者收益"].cumsum()

            payback_row = df[df["累计投资者收益"] >= one_time_cost].head(1)
            if not payback_row.empty:
                return (payback_row["场次时间"].values[0] - pd.to_datetime(start_date)).days
        except:
            return None
        return None

    if len(selected_optimizable) != 1:
        return {"⚠️ 参数选择错误": "一次只能选择一个优化参数，请重新选择"}

    param = selected_optimizable[0]

    if param == "最高价格":
        current_price = max_price
        best_result = None
        best_price = None
        closest_diff = None

        for price in range(int(current_price) + 20, int(current_price * 2) + 1, 20):
            result = simulate(input_dict, tag_values, start_date, end_date, weekly_plan, price, price * 0.5)
            if result and result <= target_days:
                diff = abs(result - target_days)
                if closest_diff is None or diff < closest_diff:
                    best_result = result
                    best_price = price
                    closest_diff = diff

        if best_result is not None:
            suggestions["最高价格"] = f"建议提高至 {best_price} 元（投资者回本周期：{best_result} 天）"
            return suggestions

    elif param == "周期":
        current_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        best_result = None
        best_days = None
        closest_diff = None

        for extra_days in range(30, 181, 30):
            new_end = pd.to_datetime(start_date) + pd.Timedelta(days=current_days + extra_days)
            result = simulate(input_dict, tag_values, start_date, new_end, weekly_plan, max_price, min_price)
            if result and result <= target_days:
                diff = abs(result - target_days)
                if closest_diff is None or diff < closest_diff:
                    best_result = result
                    best_days = (new_end - pd.to_datetime(start_date)).days
                    closest_diff = diff

        if best_result is not None:
            suggestions["周期"] = f"建议延长至 {best_days} 天（投资者回本周期：{best_result} 天）"
            return suggestions

    elif param == "是否常驻":
        best_result = None
        best_val = None
        closest_diff = None

        for val in [0, 1]:
            if val == input_dict["是否常驻"]:
                continue
            new_input = input_dict.copy()
            new_input["是否常驻"] = val
            result = simulate(new_input, tag_values, start_date, end_date, weekly_plan, max_price, min_price)
            if result and result <= target_days:
                diff = abs(result - target_days)
                if closest_diff is None or diff < closest_diff:
                    best_result = result
                    best_val = val
                    closest_diff = diff

        if best_result is not None:
            suggestions["是否常驻"] = f"建议设为 {'是' if best_val == 1 else '否'}（投资者回本周期：{best_result} 天）"
            return suggestions

    elif param == "剧场规模":
        best_result = None
        best_val = None
        closest_diff = None

        for val in [0, 1]:
            if val == input_dict["剧场规模"]:
                continue
            new_input = input_dict.copy()
            new_input["剧场规模"] = val
            result = simulate(new_input, tag_values, start_date, end_date, weekly_plan, max_price, min_price)
            if result and result <= target_days:
                diff = abs(result - target_days)
                if closest_diff is None or diff < closest_diff:
                    best_result = result
                    best_val = val
                    closest_diff = diff

        if best_result is not None:
            suggestions["剧场规模"] = f"建议使用 {'大中剧场' if best_val == 1 else '小剧场'}（投资者回本周期：{best_result} 天）"
            return suggestions

    elif param == "剧场区域":
        best_result = None
        best_val = None
        closest_diff = None

        for val in region_map.values():
            if val == input_dict["剧场区域"]:
                continue
            new_input = input_dict.copy()
            new_input["剧场区域"] = val
            result = simulate(new_input, tag_values, start_date, end_date, weekly_plan, max_price, min_price)
            if result and result <= target_days:
                diff = abs(result - target_days)
                if closest_diff is None or diff < closest_diff:
                    best_result = result
                    best_val = val
                    closest_diff = diff

        if best_result is not None:
            suggestions["剧场区域"] = f"建议调整为 {reverse_region_map[best_val]}（投资者回本周期：{best_result} 天）"
            return suggestions

    elif param == "题材标签":
        best_result = None
        best_tag = None
        closest_diff = None

        for tag, val in tag_values.items():
            if val == 1:
                continue
            new_tags = tag_values.copy()
            new_tags[tag] = 1
            result = simulate(input_dict, new_tags, start_date, end_date, weekly_plan, max_price, min_price)
            if result and result <= target_days:
                diff = abs(result - target_days)
                if closest_diff is None or diff < closest_diff:
                    best_result = result
                    best_tag = tag
                    closest_diff = diff

        if best_result is not None:
            suggestions[f"题材标签：{best_tag}"] = f"建议添加该标签（投资者回本周期：{best_result} 天）"
            return suggestions

    suggestions["❌ 无法优化"] = "在当前参数范围内无法实现目标投资者回本周期"
    return suggestions



st.set_page_config(layout="wide")
st.markdown("""
    <style>
    /* 不限制外层 wrapper 高度 */
    div[data-testid="stExpander"] > details > summary + div {
        overflow: visible !important;
    }

    /* 不限制 wrapper */
    div[data-testid="stExpander"] > details > summary + div > div {
        overflow: visible !important;
    }

    /* ✅ 限制真正的内容区域高度 */
    div[data-testid="stExpander"] .stExpanderContent {
        max-height: 300px;
        overflow-y: auto;
        padding-right: 0.5rem;
        box-sizing: border-box;
    }
    </style>
""", unsafe_allow_html=True)




st.title("🎭 剧目营收预测系统")

def collect_cost_inputs():
    st.markdown("## 💰 成本参数设置")
    st.markdown("### 一次性投入成本")

    # 🎬 创作类成本
    with st.expander("🎬 创作类成本", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            cost_copyright = st.number_input("版权使用费", value=0)
            cost_director = st.number_input("导演", value=75000)
            cost_writer = st.number_input("编剧/作词", value=75000)
            cost_music = st.number_input("音乐创作及编曲", value=75000)
            cost_recording = st.number_input("音乐录制", value=75000)
        with col2:
            cost_costume = st.number_input("服装设计与制作", value=50000)
            cost_light = st.number_input("灯光设计与落地", value=30000)
            cost_choreography = st.number_input("编舞设计", value=20000)
            cost_stage = st.number_input("剧场及舞美设计", value=50000)
            cost_audio = st.number_input("音效设计", value=30000)

    # 🎭 舞美与技术类成本
    with st.expander("🎭 舞美与技术类成本", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            cost_acoustics = st.number_input("声场声效设计", value=50000)
            cost_multimedia = st.number_input("多媒体设计", value=0)
            cost_visual = st.number_input("视觉设计", value=20000)
            cost_equipment = st.number_input("灯音麦等设备", value=380000)
            cost_emergency1 = st.number_input("应急预算（创作）", value=50000)
        with col2:
            cost_tech = st.number_input("技术执行", value=25000)
            cost_manager_big = st.number_input("大舞监（含行政运营）", value=100000)
            cost_manager_small = st.number_input("小舞监", value=40000)

    # 🏗️ 场地与基础设施成本
    with st.expander("🏗️ 场地与基础设施成本", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            cost_fire = st.number_input("消防设计+施工", value=300000)
            cost_hard = st.number_input("硬装设计+施工+监控", value=380000)
            cost_soft = st.number_input("舞美软装道具+化妆间+吧台", value=380000)
        with col2:
            cost_network = st.number_input("宽带网络", value=10000)
            cost_property = st.number_input("物业费", value=150000)
            cost_emergency2 = st.number_input("应急预算（运营）", value=100000)

    # 👥 人员与排练成本
    with st.expander("👥 人员与排练成本", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            cost_rehearsal = st.number_input("排练费", value=75000)
            cost_rehearsal_place = st.number_input("排练场地", value=30000)
            cost_kami = st.number_input("卡米工资", value=240000)
        with col2:
            cost_emergency3 = st.number_input("应急预算（其他）", value=30000)

    # 📣 宣发与运营成本
    with st.expander("📣 宣发与运营成本", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            cost_material = st.number_input("剧目宣发及物料制作", value=10000)
            cost_ticketing = st.number_input("票务平台", value=10000)
            cost_marketing = st.number_input("宣发营销", value=10000)
        with col2:
            cost_emergency4 = st.number_input("应急预算（宣传）", value=10000)
            cost_operation = st.number_input("运营投入", value=700000)

    one_time_cost = sum([
        cost_copyright, cost_director, cost_writer, cost_music, cost_recording,
        cost_costume, cost_light, cost_choreography, cost_stage, cost_audio,
        cost_acoustics, cost_multimedia, cost_visual, cost_equipment, cost_emergency1,
        cost_tech, cost_manager_big, cost_manager_small,
        cost_fire, cost_hard, cost_soft, cost_network, cost_property, cost_emergency2,
        cost_rehearsal, cost_rehearsal_place, cost_kami, cost_emergency3,
        cost_material, cost_ticketing, cost_marketing, cost_emergency4,
        cost_operation
    ])

    # 🎟️ 持续性投入成本
    with st.expander("🎟️ 持续性投入成本（单位：元/场）", expanded=False):
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

    # 🧾 管理费用
    with st.expander("🧾 管理费用（按月）", expanded=False):
        col_admin1, col_admin2 = st.columns(2)
        with col_admin1:
            monthly_admin_fixed = st.number_input("管理费用（固定）", value=120000)
        with col_admin2:
            monthly_property = st.number_input("物业费用", value=0)

        monthly_admin = monthly_admin_fixed + monthly_property

        return one_time_cost, per_show_cost, monthly_admin


    
def collect_distribution_inputs():
    with st.expander("📊 收入分成参数设置", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            venue_share = st.number_input("场地分成（%）", value=5.0) / 100
        with col2:
            tax_rate = st.number_input("税点（%）", value=3.0) / 100
        with col3:
            channel_share = st.number_input("票房渠道分成（%）", value=14.0) / 100

        col4, col5 = st.columns(2)
        with col4:
            investor_share_payback = st.number_input("投资者分成占比（回本期 %）", value=50.0) / 100
        with col5:
            investor_share_profit = st.number_input("投资者分成占比（收益期 %）", value=20.0) / 100

    return venue_share, tax_rate, channel_share, investor_share_payback, investor_share_profit




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

    # 定义特征权重矩阵
    def get_feature_weights(tag_values):
        tag_cols = list(tag_values.keys())
        return {
            "通用模型": {},
            "运营侧重模型": {
                "最高价格": 1.5,
                "最低价格": 1.3,
                "营销程度": 1.5,
                "周期": 1.0,
                "总座位数": 1.0
            },
            "内容侧重模型": {
                "演员阵容": 1.8,
                "互动指数": 1.3,
                **{tag: 1.8 for tag in tag_cols}
            },
            "竞争侧重模型": {
                "是否节假日": 1.5,
                "是否下午场": 1.2,
                "是否周末": 1.4,
                "竞争程度": 1.5
            },
            "区域及排期侧重模型": {
                "剧场区域": 1.5,
                "剧目类型": 1.3,
                "周期": 1.3,
                "是否常驻": 1.2,
                "剧场规模": 1.2,
                "总座位数": 1.2
            }
        }


    
    def suggest_model_type(input_dict, tag_values, marketing_level, competition_level):
        reasons = []
        tag_score = sum(tag_values.values())
        actor_score = input_dict.get("演员阵容", 0)
        interaction_score = input_dict.get("互动指数", 0)
        duration = input_dict.get("周期", 30)
        resident = input_dict.get("是否常驻", 0)
        scale = input_dict.get("剧场规模", 0)
        region = input_dict.get("剧场区域", 0)

        # 评分逻辑
        if marketing_level >= 20 or max_price >= 600:
            reasons.append("营销程度较高，适合运营侧重模型")
            return "运营侧重模型", reasons
        elif actor_score >= 3 or tag_score >= 3 or interaction_score >= 4:
            reasons.append("演员阵容强或题材丰富，适合内容侧重模型")
            return "内容侧重模型", reasons
        elif competition_level >= 3:
            reasons.append("竞争程度较高，适合竞争侧重模型")
            return "竞争侧重模型", reasons
        elif duration >= 180 or resident == 1 or scale == 1:
            reasons.append("周期较长或常驻/大剧场，适合区域及排期侧重模型")
            return "区域及排期侧重模型", reasons
        else:
            reasons.append("参数特征较均衡，适合通用模型")
            return "通用模型", reasons




    # one-hot 编码（自动处理分类变量）
    X = pd.get_dummies(X_raw)

    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y_raw, test_size=0.2, random_state=42)

    # 模型选择
    model_options = ["Random Forest", "LightGBM"]
    model_name = st.selectbox("选择模型", model_options)

    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "LightGBM":
        model = LGBMRegressor(n_estimators=100, random_state=42)

    #elif model_name == "Ridge Regression":
        #model = Ridge()
    #elif model_name == "XGBoost":
        #model = XGBRegressor(n_estimators=100, random_state=42)
    #elif model_name == "MLP (多层感知机)":
        #model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

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
    # 初始化 session_state 控制预测执行
    if "run_prediction" not in st.session_state:
        st.session_state.run_prediction = False

    # 🆕 输入新剧信息进行预测
    with st.expander("🆕 输入新剧信息进行预测", expanded=True):

        # 🎭 剧目参数设置（参与模型预测）
        st.markdown("### 🧩 参数设置（按类型分组）")

        # === 🎭 其他参数 ===
        with st.expander("🎭 其他参数", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                show_type = st.selectbox("剧目类型", list(type_map.keys()))
            with col2:
                is_resident = st.selectbox("是否常驻", list(resident_map.keys()))
            with col3:
                scale = st.selectbox("剧场规模", list(scale_map.keys()))
            region = st.selectbox("剧场区域", list(region_map.keys()))
            seat_count = st.number_input("总座位数", min_value=0, value=150)

        # === 🚀 演出周期 ===
        with st.expander("📅 演出周期设置", expanded=True):
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

        # === 🗓 每周排期 ===
        with st.expander("🗓 每周排期设置", expanded=False):
            weekday_map = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"}
            time_options = ["不演", "14:30", "19:30", "14:30 和 19:30"]
            weekly_plan = {}
            for i in range(7):
                default_choice = "19:30" if i < 5 else "14:30 和 19:30"
                choice = st.selectbox(f"{weekday_map[i]}", time_options, index=time_options.index(default_choice), key=f"weekday_{i}")
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

        # === 🧠 内容参数 ===
        with st.expander("🧠 内容参数", expanded=True):
            all_tags = ["悬疑", "推理", "喜剧", "恐怖", "惊悚", "犯罪", "爱情", "历史", "传记",
                        "科幻", "奇幻", "玄幻", "灾难", "社会现实", "家庭伦理", "艺术文化", "战争", "职场", "其他"]
            selected_tags = st.multiselect("请选择题材标签（可多选）", options=all_tags)
            tag_values = {tag: (1 if tag in selected_tags else 0) for tag in all_tags}

            col1, col2 = st.columns(2)
            with col1:
                actor_count = st.number_input("演员阵容（知名演员数量）", min_value=0, value=3)
            with col2:
                interaction_score = st.slider("互动指数（0-5）", min_value=0.0, max_value=5.0, step=0.1, value=3.0)

        # === 🌐 外部参数 ===
        with st.expander("🌐 外部参数", expanded=True):
            competition_level = st.number_input("竞争程度（同期竞品数量）", min_value=0, value=2)

        # === 📣 运营参数 ===
        with st.expander("📣 运营参数", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                max_price = st.number_input("最高票价", value=580)
            with col2:
                min_price = st.number_input("最低票价", value=180)
            marketing_level = st.number_input("营销程度（搜索热度）", min_value=0, value=15)

    
        # 💰 成本参数设置（不参与模型预测）
        st.markdown("### 💰 成本参数设置（仅用于收益分析）")
        one_time_cost, per_show_cost, monthly_admin = collect_cost_inputs()

        # 获取分成参数
        venue_share, tax_rate, channel_share, investor_share_payback, investor_share_profit = collect_distribution_inputs()

        input_dict = {
            "剧目类型": type_map[show_type],
            "是否常驻": resident_map[is_resident],
            "剧场规模": scale_map[scale],
            "剧场区域": region_map[region],
            "演员阵容": actor_count,
            "互动指数": interaction_score,
            "营销程度": marketing_level,
            "竞争程度": competition_level,
            "总座位数": seat_count
        }

        # 自动推荐模型类型
        auto_model_type, auto_reasons = suggest_model_type(
            input_dict=input_dict,
            tag_values=tag_values,
            marketing_level=marketing_level,
            competition_level=competition_level
        )

        # 模型维度选择
        st.markdown("### 🧠 特征关注模型选择")
        model_types = ["通用模型", "运营侧重模型", "内容侧重模型", "竞争侧重模型", "区域及排期侧重模型"]
        selected_model_type = st.selectbox("选择特征关注模型", model_types, index=model_types.index(auto_model_type))
        st.markdown("### 🤖 推荐模型类型")
        st.success(f"系统推荐使用模型：**{auto_model_type}**")
        for reason in auto_reasons:
            st.markdown(f"- {reason}")


    
        # 🚀 开始预测
        if st.button("开始预测"):
            st.session_state.run_prediction = True

        
        # 初始化权重配置
        feature_weights_all = get_feature_weights(tag_values)

        # 显示特征权重滑块（不显示具体数值）
        

        raw_default_weights = feature_weights_all.get(selected_model_type, {})
        default_weights = {col: raw_default_weights.get(col, 1.0) for col in X.columns}

        # 🎛 特征权重调整（按图示分组）
        # 🎛 特征权重调整（按图示分组）
        st.markdown("🎛 特征权重调整")

        adjusted_weights = {}
        already_handled = set()

        # 第一行：运营参数 + 内容参数
        col1, col2 = st.columns(2)

        with col1:
            with st.expander("📣 运营参数", expanded=True):
                for feature in ["最高价格", "最低价格", "营销程度", "周期", "总座位数"]:
                    if feature in X.columns:
                        default = default_weights.get(feature, 1.0)
                        weight = st.slider(feature, 0.0, 3.0, step=0.1, value=default, key=f"slider_{feature}")
                        adjusted_weights[feature] = weight
                        already_handled.add(feature)

        with col2:
            with st.expander("🎭 内容参数", expanded=True):
                sample_tag = next((tag for tag in tag_values if tag in default_weights), None)
                tag_default = default_weights.get(sample_tag, 1.0) if sample_tag else 1.0
                tag_weight = st.slider("题材标签", 0.0, 3.0, step=0.1, value=tag_default, key="slider_题材标签")
                for tag in tag_values.keys():
                    adjusted_weights[tag] = tag_weight
                    already_handled.add(tag)

                for feature in ["演员阵容", "互动指数"]:
                    if feature in X.columns:
                        default = default_weights.get(feature, 1.0)
                        weight = st.slider(feature, 0.0, 3.0, step=0.1, value=default, key=f"slider_{feature}")
                        adjusted_weights[feature] = weight
                        already_handled.add(feature)

        # 第二行：外部参数 + 其他参数
        col3, col4 = st.columns(2)

        with col3:
            with st.expander("🌐 外部参数", expanded=True):
                for feature in ["竞争程度", "是否节假日", "是否周末", "是否下午场"]:
                    if feature in X.columns:
                        default = default_weights.get(feature, 1.0)
                        weight = st.slider(feature, 0.0, 3.0, step=0.1, value=default, key=f"slider_{feature}")
                        adjusted_weights[feature] = weight
                        already_handled.add(feature)

        with col4:
            with st.expander("🧩 其他参数", expanded=True):
                for feature in X.columns:
                    if feature in already_handled:
                        continue
                    if "_" in feature and any(feature.startswith(prefix + "_") for prefix in ["剧场区域", "剧目类型"]):
                        continue
                    default = default_weights.get(feature, 1.0)
                    weight = st.slider(feature, 0.0, 3.0, step=0.1, value=default, key=f"slider_{feature}")
                    adjusted_weights[feature] = weight
                    already_handled.add(feature)

        # 更新当前模型类型对应的权重
        feature_weights_all[selected_model_type] = adjusted_weights



        if st.session_state.run_prediction:

    
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
            schedule_df["演出月份"] = schedule_df["场次时间"].dt.month
    
            for k, v in input_dict.items():
                schedule_df[k] = v
            for tag, val in tag_values.items():
                schedule_df[tag] = val
    
            # one-hot 编码
            X_new = pd.get_dummies(schedule_df.drop(columns=["场次时间"]))
            def apply_feature_weights(X, weight_dict):
                X_weighted = X.copy()
                for feature, weight in weight_dict.items():
                    if feature in X_weighted.columns:
                        X_weighted[feature] *= weight
                return X_weighted

            # 应用权重
            X_new = apply_feature_weights(X_new, feature_weights_all[selected_model_type])

            X_new = X_new.reindex(columns=X.columns, fill_value=0)
    
            # 模型预测
            try:
                y_new = model.predict(X_new)
                schedule_df["预测营收"] = y_new
    
                # 📊 可视化
                # 添加预测营收
                # 营收扣除场地、税、渠道分成
                net_ratio = 1 - venue_share - tax_rate - channel_share
                schedule_df["预测营收"] = y_new * net_ratio

                
                # 计算累计营收
                schedule_df["累计预测营收"] = schedule_df["预测营收"].cumsum()
                
                # 计算成本
                num_shows = len(schedule_df)
                period = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
                admin_cost = monthly_admin * (period / 30)
                
                # 每场管理成本（平均分摊）
                admin_per_show = admin_cost / num_shows
                schedule_df["累计成本"] = one_time_cost + per_show_cost * np.arange(1, num_shows + 1) + admin_per_show * np.arange(1, num_shows + 1)
                
                # 每场收益、累计收益
                schedule_df["每场收益"] = schedule_df["预测营收"] - (per_show_cost + admin_per_show)
                schedule_df["累计收益"] = schedule_df["每场收益"].cumsum()
               # 投资者 vs 运营者收益拆分
                investor_share = []
                operator_share = []
                cumulative_profit = 0
                
                for i, profit in enumerate(schedule_df["每场收益"]):
                    cumulative_profit += profit
                    if cumulative_profit < one_time_cost:
                        investor_ratio = investor_share_payback
                    else:
                        investor_ratio = investor_share_profit
                    investor_share.append(profit * investor_ratio)
                    operator_share.append(profit * (1 - investor_ratio))
                
                schedule_df["投资者收益"] = investor_share
                schedule_df["运营者收益"] = operator_share
                schedule_df["累计投资者收益"] = schedule_df["投资者收益"].cumsum()
                schedule_df["累计运营者收益"] = schedule_df["运营者收益"].cumsum()

                st.info(f"📌 当前使用的特征关注模型：**{selected_model_type}**")

                # 图 1：每场预测营收（条形图）
                st.subheader("📊 每场预测营收（条形图）")
                fig1, ax1 = plt.subplots(figsize=(12, 5))
                ax1.bar(schedule_df["场次时间"], schedule_df["预测营收"], color="#2196F3")
                ax1.set_title("每场次预测营收")
                ax1.set_xlabel("场次时间")
                ax1.set_ylabel("预测营收（元）")
                ax1.tick_params(axis='x', rotation=45)
                st.pyplot(fig1)
                
                # 图 2：累计营收 vs 累计成本（折线图）
                st.subheader("📈 累计营收 vs 累计成本")
                fig2, ax2 = plt.subplots(figsize=(12, 5))
                ax2.plot(schedule_df["场次时间"], schedule_df["累计预测营收"], marker='o', label="累计预测营收", color="#2196F3")
                ax2.plot(schedule_df["场次时间"], schedule_df["累计成本"], marker='s', label="累计成本", color="#FF5722")
                ax2.set_title("累计营收 vs 累计成本")
                ax2.set_xlabel("场次时间")
                ax2.set_ylabel("金额（元）")
                ax2.legend()
                ax2.grid(True)
                ax2.tick_params(axis='x', rotation=45)
                st.pyplot(fig2)
                
               # 图 3：投资者收益
                st.subheader("💹 投资者收益趋势")
                fig3, ax3 = plt.subplots(figsize=(12, 5))
                ax3.bar(schedule_df["场次时间"], schedule_df["投资者收益"], label="每场投资者收益", color="#FF9800")
                ax3.plot(schedule_df["场次时间"], schedule_df["累计投资者收益"], label="累计投资者收益", color="#E65100", marker='o')
                ax3.set_ylabel("金额（元）")
                ax3.set_title("投资者收益趋势")
                ax3.legend()
                ax3.tick_params(axis='x', rotation=45)
                st.pyplot(fig3)
                
                # 图 4：运营者收益
                st.subheader("💹 运营者收益趋势")
                fig4, ax4 = plt.subplots(figsize=(12, 5))
                ax4.bar(schedule_df["场次时间"], schedule_df["运营者收益"], label="每场运营者收益", color="#4CAF50")
                ax4.plot(schedule_df["场次时间"], schedule_df["累计运营者收益"], label="累计运营者收益", color="#1B5E20", marker='s')
                ax4.set_ylabel("金额（元）")
                ax4.set_title("运营者收益趋势")
                ax4.legend()
                ax4.tick_params(axis='x', rotation=45)
                st.pyplot(fig4)


    
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
                # 回本周期计算
                payback_row = schedule_df[schedule_df["累计预测营收"] >= schedule_df["累计成本"]].head(1)
                if not payback_row.empty:
                    payback_date = payback_row["场次时间"].values[0]
                    payback_days = (pd.to_datetime(payback_date) - pd.to_datetime(start_date)).days
                    st.markdown(f"- 回本周期：**第 {payback_days} 天（{pd.to_datetime(payback_date).date()}）** 实现盈亏平衡")
                else:
                    st.markdown("- 回本周期：**未在预测周期内实现盈亏平衡**")
                # 投资者回本周期
                investor_payback_row = schedule_df[schedule_df["累计投资者收益"] >= one_time_cost].head(1)
                if not investor_payback_row.empty:
                    payback_date = investor_payback_row["场次时间"].values[0]
                    payback_days = (pd.to_datetime(payback_date) - pd.to_datetime(start_date)).days
                    st.markdown(f"- 🎯 投资者回本周期：**第 {payback_days} 天（{pd.to_datetime(payback_date).date()}）**")
                else:
                    st.markdown("- 🎯 投资者回本周期：**未在预测周期内实现回本**")

                                # 🎯 回本优化建议
                st.markdown("### 🎯 回本优化建议")
                target_days = st.number_input("请输入目标投资者回本周期（单位：天）", value=90, min_value=1)
                optimizable_options = ["最高价格", "周期", "是否常驻", "剧场规模", "剧场区域", "题材标签"]

                selected_optimizable = st.multiselect("可优化参数（一次只能选一个）", options=optimizable_options, max_selections=1)

                if st.button("📈 生成优化建议"):
                    suggestions = suggest_parameter_adjustments(
                        model, X.columns, one_time_cost, per_show_cost, monthly_admin,
                        investor_share_payback, investor_share_profit, venue_share, tax_rate, channel_share,
                        start_date, end_date, target_days,
                        input_dict=input_dict,
                        tag_values=tag_values,
                        selected_optimizable=selected_optimizable,
                        weekly_plan=weekly_plan,
                        holiday_list=holiday_list,
                        max_price=max_price,
                        min_price=min_price
                    )

                    if suggestions:
                        st.info("📌 以下是可供参考的参数优化建议，以实现目标回本周期：")
                        for k, v in suggestions.items():
                            st.markdown(f"- **{k}**：{v}")
                    else:
                        st.warning("⚠️ 无法在当前参数范围内提供可行的优化建议")

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
