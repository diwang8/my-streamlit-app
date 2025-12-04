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

        # 绘图
        st.subheader("📈 场次营收预测")
        selected_rows["预测营收"] = y_pred
        selected_rows = selected_rows.sort_values("场次时间")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(selected_rows["场次时间"], selected_rows["营业收入"], marker='o', label="实际营收", color="#4CAF50")
        ax.plot(selected_rows["场次时间"], selected_rows["预测营收"], marker='o', label="预测营收", color="#2196F3")
        ax.set_title(f"{selected_name} 每场次营收对比")
        ax.set_xlabel("场次时间")
        ax.set_ylabel("营收")
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")

    # 🆕 输入新剧信息进行预测
    with st.expander("🆕 输入新剧信息进行预测"):
        # 成本输入
        one_time_cost, per_show_cost, monthly_admin = collect_cost_inputs()

        col1, col2 = st.columns(2)
        with col1:
            type_text = st.selectbox("类型", list(type_map.keys()))
            min_price = st.number_input("最低价格", value=100)
            max_price = st.number_input("最高价格", value=500)
            period = st.number_input("周期（天）", value=30)
            resident_text = st.selectbox("是否常驻", list(resident_map.keys()))
        with col2:
            scale_text = st.selectbox("剧场规模", list(scale_map.keys()))
            region_text = st.selectbox("剧场区域", list(region_map.keys()))
            tags = st.multiselect("题材标签", [
                "悬疑", "推理", "喜剧", "恐怖", "惊悚", "犯罪", "爱情", "历史", "传记", "奇幻", "玄幻",
                "灾难", "社会现实", "家庭伦理", "艺术文化", "战争", "职场"
            ])

        input_dict = {
            "类型": type_map[type_text],
            "最低价格": min_price,
            "最高价格": max_price,
            "周期": period,
            "是否常驻": resident_map[resident_text],
            "剧场规模": scale_map[scale_text],
            "剧场区域": str(region_map[region_text])
        }
        for tag in [
            "悬疑", "推理", "喜剧", "恐怖", "惊悚", "犯罪", "爱情", "历史", "传记", "奇幻", "玄幻",
            "灾难", "社会现实", "家庭伦理", "艺术文化", "战争", "职场"
        ]:
            input_dict[tag] = 1 if tag in tags else 0

        input_df = pd.DataFrame([input_dict])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        if st.button("🚀 预测新剧营收"):
            pred = model.predict(input_df)[0]

    # 初始化 session_state
            if "last_pred" not in st.session_state:
                st.session_state.last_pred = None
            if "last_input" not in st.session_state:
                st.session_state.last_input = None

            # 显示输入参数对比
            st.subheader("📋 输入参数对比")
            current_input_display = input_df.copy()
            last_input_display = pd.DataFrame(st.session_state.last_input) if st.session_state.last_input is not None else None

            if last_input_display is not None:
                compare_df = pd.concat([last_input_display.T, current_input_display.T], axis=1)
                compare_df.columns = ["上一次输入", "本次输入"]
                st.dataframe(compare_df)
            else:
                st.dataframe(current_input_display.T.rename(columns={0: "本次输入"}))

            # 显示预测结果
            st.subheader("📈 预测结果")
            if predict_average:
                st.metric("预测场均营收", f"{pred:.2f} 元")

                # 仅当上一次预测是标量时才绘图
                if st.session_state.last_pred is not None and np.isscalar(st.session_state.last_pred):
                    fig, ax = plt.subplots()
                    ax.bar(["上一次预测", "本次预测"], [st.session_state.last_pred, pred], color=["#FF9800", "#2196F3"])
                    ax.set_title("场均营收预测对比")
                    ax.set_ylabel("营收")
                    st.pyplot(fig)
            else:
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                ax[0].bar(range(1, 22), pred, color="#2196F3", label="本次预测")
                if st.session_state.last_pred is not None and isinstance(st.session_state.last_pred, (list, np.ndarray)):
                    ax[0].bar(range(1, 22), st.session_state.last_pred, color="#FF9800", alpha=0.5, label="上一次预测")
                ax[0].set_title("每场营收预测对比")
                ax[0].set_xlabel("场次")
                ax[0].set_ylabel("营收")
                ax[0].legend()

                ax[1].plot(np.cumsum(pred), marker='o', label="本次预测", color="#2196F3")
                if st.session_state.last_pred is not None and isinstance(st.session_state.last_pred, (list, np.ndarray)):
                    ax[1].plot(np.cumsum(st.session_state.last_pred), marker='o', label="上一次预测", color="#FF9800")
                ax[1].set_title("累计营收预测对比")
                ax[1].set_xlabel("场次")
                ax[1].set_ylabel("累计营收")
                ax[1].legend()
                st.pyplot(fig)
            # 计算收益
            num_shows = 21 if not predict_average else 1
            admin_cost = monthly_admin * (period / 30)
            recurring_cost = per_show_cost * num_shows
            total_cost = one_time_cost + recurring_cost + admin_cost

            st.subheader("💵 成本与收益分析")
            st.markdown(f"- 一次性投入成本：**{one_time_cost:,.2f} 元**")
            st.markdown(f"- 每场成本：**{per_show_cost:,.2f} 元** × {num_shows} 场 = {recurring_cost:,.2f} 元")
            st.markdown(f"- 管理费用：**{monthly_admin:,.2f} 元/月** × {period} 天 ≈ {admin_cost:,.2f} 元")
            st.markdown(f"### ✅ 总成本：**{total_cost:,.2f} 元**")

            if not predict_average:
                profit = pred - per_show_cost
                cum_profit = np.cumsum(profit)

                fig, ax = plt.subplots(1, 2, figsize=(14, 5))
                ax[0].bar(np.arange(1, 22), pred, label="营收", color="#2196F3")
                ax[0].bar(np.arange(1, 22), [per_show_cost]*21, label="成本", color="#FF9800", alpha=0.6)
                ax[0].bar(np.arange(1, 22), profit, label="收益", color="#4CAF50", alpha=0.6)
                ax[0].set_title("每场营收 / 成本 / 收益")
                ax[0].legend()

                ax[1].plot(np.arange(1, 22), cum_profit, marker='o', label="累计收益", color="#4CAF50")
                ax[1].axhline(y=total_cost, color='red', linestyle='--', label="总成本")
                ax[1].set_title("累计收益曲线")
                ax[1].legend()

                st.pyplot(fig)

            # 保存当前输入和预测
            st.session_state.last_input = input_df.to_dict(orient="records")
            st.session_state.last_pred = float(pred) if predict_average else np.array(pred)

            # 导出结果
            st.subheader("💾 导出预测结果")
            export_df = input_df.copy()
            if predict_average:
                export_df["预测场均营收"] = pred
            else:
                for i in range(21):
                    export_df[f"第{i+1}场预测营收"] = pred[i]
                export_df["累计预测营收"] = np.sum(pred)

            csv = export_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 下载预测结果 CSV",
                data=csv,
                file_name="预测结果.csv",
                mime="text/csv"
            )





















