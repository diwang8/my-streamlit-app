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
uploaded_file = st.file_uploader("📤 上传剧目信息数据文件（CSV）", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 选择是否只预测场均营收
    predict_average = st.checkbox("✅ 只预测场均营收")

    # 自动识别营收列（包含“第”和“场营收”）
    revenue_cols = [col for col in df.columns if "第" in col and "场营收" in col]

    # 自动识别特征列（排除剧目名称和营收列）
    exclude_cols = ["剧目名称"] + revenue_cols
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # 构造训练数据
    X_raw = df[feature_cols].copy()
    y_raw = df[revenue_cols].copy().fillna(0)

    if predict_average:
        y_raw = y_raw.mean(axis=1)  # Series

    # 自动映射字段（如类型、是否常驻等）
    mapping_fields = {
        "类型": type_map,
        "是否常驻": {"否": 0, "是": 1, "N": 0, "Y": 1},
        "剧场规模": scale_map,
        "剧场区域": region_map
    }
    for col, mapping in mapping_fields.items():
        if col in X_raw.columns:
            X_raw[col] = X_raw[col].map(mapping).fillna(X_raw[col])

    # 自动识别分类字段（非数值型）
    categorical_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        X_raw[col] = X_raw[col].astype(str)

    # one-hot 编码
    X = pd.get_dummies(X_raw)


    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y_raw, test_size=0.2, random_state=42)

    # 模型选择
    if predict_average:
        model_options = ["Random Forest", "Ridge Regression", "XGBoost", "LightGBM", "MLP (多层感知机)"]
    else:
        model_options = ["Random Forest", "Ridge Regression", "LightGBM", "MLP (多层感知机)"] 
    model_name = st.selectbox("选择模型", model_options)

    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "Ridge Regression":
        model = Ridge()
    elif model_name == "XGBoost":
        base_model = XGBRegressor(n_estimators=100, random_state=42)
        model = base_model if predict_average else MultiOutputRegressor(base_model)
    elif model_name == "LightGBM":
        base_model = LGBMRegressor(n_estimators=100, random_state=42)
        model = base_model if predict_average else MultiOutputRegressor(base_model)
    elif model_name == "MLP (多层感知机)":
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

    # 训练模型
    model.fit(X_train, y_train)

    # 模型评分
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    # st.success(f"模型 R² 分数：{score:.4f}")

    st.markdown("---")

    # 🎯 预测已有剧目
    st.subheader("🎯 选择已有剧目进行预测")
    selected_name = st.selectbox("选择剧目", df["剧目名称"].unique())
    selected_row = df[df["剧目名称"] == selected_name]

    if not selected_row.empty:
        info = selected_row.iloc[0]
        st.markdown(f"""
        - **剧目名称**: {info['剧目名称']}
        - **类型**: {reverse_type_map.get(info['类型'], info['类型'])}
        - **最低价格**: {info['最低价格']} 元
        - **最高价格**: {info['最高价格']} 元
        - **周期**: {info['周期']} 天
        - **是否常驻**: {reverse_resident_map.get(info['是否常驻'], info['是否常驻'])}
        - **剧场规模**: {reverse_scale_map.get(info['剧场规模'], info['剧场规模'])}
        - **剧场区域**: {reverse_region_map.get(info['剧场区域'], info['剧场区域'])}
        """)

        input_data = selected_row[feature_cols].copy()
        input_data["剧场区域"] = input_data["剧场区域"].astype(str)
        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=X.columns, fill_value=0)
        prediction = model.predict(input_data)[0]

        if predict_average:
            actual_avg = selected_row[revenue_cols].values.flatten().mean()
            st.metric("预测场均营收", f"{prediction:.2f} 元")
            st.metric("实际场均营收", f"{actual_avg:.2f} 元")
            fig, ax = plt.subplots()
            ax.bar(["实际值", "预测值"], [actual_avg, prediction], color=["#4CAF50", "#2196F3"])
            ax.set_ylabel("场均营收")
            ax.set_title("场均营收对比")
            st.pyplot(fig)
        else:
            actual_values = selected_row[revenue_cols].values.flatten()
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))

            ax[0].bar(np.arange(1, 22) - 0.2, actual_values, width=0.4, label="实际", color="#4CAF50")
            ax[0].bar(np.arange(1, 22) + 0.2, prediction, width=0.4, label="预测", color="#2196F3")
            ax[0].set_title("每场营收对比")
            ax[0].set_xlabel("场次")
            ax[0].set_ylabel("营收")
            ax[0].legend()

            ax[1].plot(np.arange(1, 22), np.cumsum(actual_values), marker='o', label="实际", color="#4CAF50")
            ax[1].plot(np.arange(1, 22), np.cumsum(prediction), marker='o', label="预测", color="#2196F3")
            ax[1].set_title("累计营收对比")
            ax[1].set_xlabel("场次")
            ax[1].set_ylabel("累计营收")
            ax[1].legend()

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



















