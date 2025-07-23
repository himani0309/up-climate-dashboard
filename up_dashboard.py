import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="UP Climate-Yield Dashboard", layout="wide")

# ---------------- Load All Data ---------------- #
@st.cache_data
def load_data():
    past_df = pd.read_excel("up_agrometero_data/final_climate_mint_dataset.xlsx")
    ssp245 = pd.read_excel("lstm_predictions_output/ssp245_predicted_yield.xlsx")
    ssp585 = pd.read_excel("lstm_predictions_output/ssp585_predicted_yield.xlsx")
    ssp245_climate = pd.read_excel("ssp245_future_climate_2024_2030_final_ready.xlsx")
    ssp585_climate = pd.read_excel("ssp585_future_climate_2024_2030_final_ready.xlsx")
    return past_df, ssp245, ssp585, ssp245_climate, ssp585_climate

past_df, ssp245_df, ssp585_df, ssp245_climate, ssp585_climate = load_data()

# ---------------- Common Seasonal Definitions ---------------- #
seasons = {
    "Winter": [1, 2, 12],
    "PreMonsoon": [3, 4, 5],
    "Monsoon": [6, 7, 8, 9],
    "PostMonsoon": [10, 11]
}
agg_methods = {
    "tmax": "mean",
    "tmin": "mean",
    "temp": "mean",
    "precip": "sum",
    "eto": "sum",
}

# ---------------- Streamlit Tabs ---------------- #
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 Historical Analysis",
    "🤖 LSTM Yield Prediction",
    "🔁 Future Correlation",
    "🎻 Seasonal Violin Plots"
])

# ---------------- Tab 1: Overview ---------------- #
with tab1:
    st.title("🌾 Uttar Pradesh Climate–Yield Dashboard")
    st.markdown("""
    This dashboard includes:
    - 📈 Historical climate-yield analysis
    - 🤖 LSTM-based yield forecasting for SSP scenarios
    - 🔍 Seasonal correlation & distribution insights
    """)

    st.subheader("🔹 Historical Dataset Preview")
    st.dataframe(past_df.head())

# ---------------- Tab 2: Historical Analysis ---------------- #
with tab2:
    st.header("📈 Historical Analysis")

    st.subheader("1. Yield vs Climate Regression")
    past_df['AvgTemp'] = past_df[[f'temp_{i}' for i in range(1, 13)]].mean(axis=1)
    past_df['MaxTemp'] = past_df[[f'tmax_{i}' for i in range(1, 13)]].mean(axis=1)
    past_df['MinTemp'] = past_df[[f'tmin_{i}' for i in range(1, 13)]].mean(axis=1)
    past_df['Precip']  = past_df[[f'precip_{i}' for i in range(1, 13)]].mean(axis=1)
    past_df['ET0']     = past_df[[f'eto_{i}' for i in range(1, 13)]].mean(axis=1)

    var_map = {
        'AvgTemp': 'Average Temperature (°C)',
        'MaxTemp': 'Maximum Temperature (°C)',
        'MinTemp': 'Minimum Temperature (°C)',
        'Precip': 'Precipitation (mm/day)',
        'ET0': 'Evapotranspiration (mm/day)'
    }

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    for i, (col, label) in enumerate(var_map.items()):
        ax = axs[i // 3][i % 3]
        sns.regplot(x=past_df[col], y=past_df['yield'], ax=ax, scatter_kws={"s": 50})
        ax.set_title(f"Yield vs {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("Yield (kg/ha)")
    fig.delaxes(axs[1, 2])
    st.pyplot(fig)

    st.subheader("2. Monthly Correlation Matrix")
    corr_matrix = past_df.drop(columns=["year", "latitude", "longitude", "location_name"]).corr()
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.subheader("3. Seasonal Correlation Heatmaps")
    def get_seasonal_data(df, season_name, months):
        seasonal = pd.DataFrame()
        seasonal["yield"] = df["yield"]
        for var, method in agg_methods.items():
            month_cols = [f"{var}_{m}" for m in months if f"{var}_{m}" in df.columns]
            if not month_cols:
                continue
            col_name = f"{var}_{season_name}_{method}"
            if method == "mean":
                seasonal[col_name] = df[month_cols].mean(axis=1)
            else:
                seasonal[col_name] = df[month_cols].sum(axis=1)
        return seasonal

    for season, months in seasons.items():
        season_df = get_seasonal_data(past_df, season, months)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(season_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.markdown(f"**{season} Season**")
        st.pyplot(fig)

    st.subheader("4. Seasonal Box Plots")
    for var in ["temp", "tmax", "tmin", "precip", "eto"]:
        df_box = pd.DataFrame()
        for season, months in seasons.items():
            cols = [f"{var}_{m}" for m in months if f"{var}_{m}" in past_df.columns]
            if cols:
                df_box[season] = past_df[cols].mean(axis=1)
        df_melt = df_box.melt(var_name="Season", value_name="Value")

        fig_bp, ax_bp = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_melt, x="Season", y="Value", palette="pastel", ax=ax_bp)
        ax_bp.set_title(f"{var.upper()} Seasonal Distribution")
        st.pyplot(fig_bp)

# ---------------- Tab 3: LSTM Prediction ---------------- #
with tab3:
    st.header("🤖 LSTM-based Yield Prediction")

    st.subheader("Predicted Yield (2024–2030)")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(ssp245_df["year"], ssp245_df["predicted_yield"], marker='o', label="SSP245")
    ax3.plot(ssp585_df["year"], ssp585_df["predicted_yield"], marker='o', label="SSP585")
    ax3.set_title("LSTM Predicted Yield")
    ax3.set_ylabel("Yield (kg/ha)")
    ax3.set_xlabel("Year")
    ax3.grid(True)
    ax3.legend()
    st.pyplot(fig3)

    st.download_button("⬇️ Download SSP245", ssp245_df.to_csv(index=False), file_name="ssp245_yield.csv")
    st.download_button("⬇️ Download SSP585", ssp585_df.to_csv(index=False), file_name="ssp585_yield.csv")

# ---------------- Tab 4: Future Correlation ---------------- #
with tab4:
    st.header("🔁 Future Correlation (SSP Scenarios)")

    def seasonal_df(full_df, predicted):
        def compute(var):
            return pd.DataFrame({
                f'{var}_PreMonsoon': full_df[[f'{var}_3', f'{var}_4', f'{var}_5']].mean(axis=1),
                f'{var}_Monsoon': full_df[[f'{var}_6', f'{var}_7', f'{var}_8', f'{var}_9']].mean(axis=1),
                f'{var}_PostMonsoon': full_df[[f'{var}_10', f'{var}_11']].mean(axis=1),
                f'{var}_Winter': full_df[[f'{var}_12', f'{var}_1', f'{var}_2']].mean(axis=1),
            })
        df = pd.concat([
            compute("temp"), compute("tmax"), compute("tmin"), compute("precip")
        ], axis=1)
        df["predicted_yield"] = predicted
        return df

    df_245 = seasonal_df(ssp245_climate, ssp245_df["predicted_yield"])
    df_585 = seasonal_df(ssp585_climate, ssp585_df["predicted_yield"])

    corr_245 = df_245.corr().loc[["predicted_yield"]].T.drop("predicted_yield")
    corr_585 = df_585.corr().loc[["predicted_yield"]].T.drop("predicted_yield")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**SSP245 Correlation**")
        fig4, ax4 = plt.subplots(figsize=(7, 6))
        sns.heatmap(corr_245, annot=True, cmap="coolwarm", center=0, ax=ax4)
        st.pyplot(fig4)

    with col2:
        st.markdown("**SSP585 Correlation**")
        fig5, ax5 = plt.subplots(figsize=(7, 6))
        sns.heatmap(corr_585, annot=True, cmap="coolwarm", center=0, ax=ax5)
        st.pyplot(fig5)

# ---------------- Tab 5: Seasonal Violin Plots ---------------- #
with tab5:
    st.header("🎻 Seasonal Distribution of Future Climate")

    def melt_variable(df, var_name):
        df_var = df[[f"{var_name}_{i}" for i in range(1, 13)]].copy()
        df_var["year"] = df["year"]
        melted = df_var.melt(id_vars="year", var_name="month", value_name=var_name)
        melted["month_num"] = melted["month"].str.extract(r'_(\d+)').astype(int)

        def month_to_season(m):
            if m in [12, 1, 2]:
                return "Winter"
            elif m in [3, 4, 5]:
                return "Pre-Monsoon"
            elif m in [6, 7, 8, 9]:
                return "Monsoon"
            else:
                return "Post-Monsoon"

        melted["season"] = melted["month_num"].apply(month_to_season)
        return melted

    for scenario_name, df_climate in zip(["SSP245", "SSP585"], [ssp245_climate, ssp585_climate]):
        st.subheader(f"{scenario_name} Seasonal Distributions")
        for var in ["temp", "tmax", "tmin", "precip"]:
            df_melt = melt_variable(df_climate, var)
            fig_vio, ax_vio = plt.subplots(figsize=(8, 5))
            sns.violinplot(data=df_melt, x="season", y=var, ax=ax_vio, inner="box", palette="Set3")
            ax_vio.set_title(f"{scenario_name} - {var.upper()} Seasonal Distribution")
            st.pyplot(fig_vio)
