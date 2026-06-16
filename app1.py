import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Streamlit Config
# ============================================================
st.set_page_config(
    page_title="Gold Price Predictor",
    page_icon="💰",
    layout="centered"
)

# ============================================================
# Title
# ============================================================
st.title("💰 Gold Price Prediction (22K & 24K)")

st.write(
    """
    This application predicts tomorrow's Gold Price using a trained
    LSTM model with calendar features such as day of week and month.
    It also provides trend analysis and visualizations.
    """
)

# ============================================================
# Load Model & Scaler
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "gold_model_with_calendar.keras"
    )

@st.cache_resource
def load_scaler():
    return joblib.load(
        "scaler_with_calendar.save"
    )

model = load_model()
scaler = load_scaler()

# ============================================================
# Upload Dataset
# ============================================================
uploaded_file = st.file_uploader(
    "📂 Upload CSV Dataset",
    type=["csv"]
)

# ============================================================
# Process Data
# ============================================================
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Check Date Column
    if "Date" not in df.columns:
        st.error("Dataset must contain a Date column.")
        st.stop()

    # Convert Date
    df["Date"] = pd.to_datetime(df["Date"])

    # Calendar Features
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.tail())

    # ========================================================
    # Feature Selection
    # ========================================================
    features = [
        "Gold_INR_g",
        "Silver_INR_g",
        "USDINR",
        "day_of_week",
        "month"
    ]

    df = df.dropna(subset=features)

    df_feat = df[features]

    scaled = scaler.transform(df_feat)

    # ========================================================
    # Gold Scaler
    # ========================================================
    gold_scaler = MinMaxScaler()

    gold_scaler.min_ = np.array([scaler.min_[0]])
    gold_scaler.scale_ = np.array([scaler.scale_[0]])

    history_len = 100
    window_size = 60
    future_days = 7

    if len(scaled) < window_size:

        st.error(
            "❌ At least 60 rows are required for prediction."
        )

    else:

        # ====================================================
        # Historical Actual Values
        # ====================================================
        y_true = df_feat["Gold_INR_g"].values[-history_len:]

        # ====================================================
        # Historical Predictions
        # ====================================================
        pred_on_hist = []

        start_idx = len(scaled) - history_len
        end_idx = len(scaled) - window_size + 1

        for i in range(start_idx, end_idx):

            window_input = scaled[i:i + window_size]

            window_input = np.expand_dims(
                window_input,
                axis=0
            )

            pred_scaled = model.predict(
                window_input,
                verbose=0
            )

            pred_on_hist.append(
                pred_scaled[0, 0]
            )

        pred_on_hist = gold_scaler.inverse_transform(
            np.array(pred_on_hist).reshape(-1, 1)
        ).flatten()

        # ====================================================
        # Future Prediction
        # ====================================================
        X_input = scaled[-window_size:]

        X_input = np.expand_dims(
            X_input,
            axis=0
        )

        future_preds = []

        current_input = X_input.copy()

        for _ in range(future_days):

            pred_scaled = model.predict(
                current_input,
                verbose=0
            )

            future_preds.append(
                pred_scaled[0, 0]
            )

            new_row = np.append(
                current_input[0, 1:, :],
                np.array(
                    [[
                        pred_scaled[0, 0],
                        current_input[0, -1, 1],
                        current_input[0, -1, 2],
                        current_input[0, -1, 3],
                        current_input[0, -1, 4]
                    ]]
                ),
                axis=0
            )

            current_input = np.expand_dims(
                new_row,
                axis=0
            )

        future_preds = gold_scaler.inverse_transform(
            np.array(future_preds).reshape(-1, 1)
        ).flatten()

        # ====================================================
        # Tomorrow Prediction
        # ====================================================
        price_24k = future_preds[0]

        price_22k = price_24k * 0.916

        st.success(
            f"📈 Predicted 24K Gold Price Tomorrow: ₹{price_24k:.2f}/gram"
        )

        st.info(
            f"💛 Predicted 22K Gold Price Tomorrow: ₹{price_22k:.2f}/gram"
        )

        # ====================================================
        # Actual vs Predicted Graph
        # ====================================================
        st.subheader("📊 Actual vs Predicted")

        x_actual = range(history_len)

        x_pred_hist = range(
            history_len - len(pred_on_hist),
            history_len
        )

        x_pred_future = range(
            history_len,
            history_len + future_days
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            x_actual,
            y_true,
            label="Actual Gold Price"
        )

        ax.plot(
            x_pred_hist,
            pred_on_hist,
            '--',
            label="Predicted History"
        )

        ax.plot(
            x_pred_future,
            future_preds,
            '--',
            label="Future Forecast"
        )

        ax.set_title(
            "Gold Price Forecast"
        )

        ax.legend()

        ax.grid(True)

        st.pyplot(fig)

        # ====================================================
        # Year-wise Trend
        # ====================================================
        st.subheader("📈 Year-wise Gold Price Trend")

        year_avg = df.groupby(
            df["Date"].dt.year
        )["Gold_INR_g"].mean()

        fig, ax = plt.subplots()

        year_avg.plot(
            marker="o",
            ax=ax
        )

        ax.set_title(
            "Year-wise Average Gold Price"
        )

        st.pyplot(fig)

        # ====================================================
        # Heatmap
        # ====================================================
        st.subheader(
            "📉 Feature Correlation Heatmap"
        )

        corr = df[
            [
                "Gold_INR_g",
                "Silver_INR_g",
                "USDINR",
                "day_of_week",
                "month"
            ]
        ].corr()

        fig, ax = plt.subplots()

        sns.heatmap(
            corr,
            annot=True,
            cmap="YlOrBr",
            ax=ax
        )

        st.pyplot(fig)

        # ====================================================
        # Last 30 Days
        # ====================================================
        st.subheader(
            "🔍 Predicted vs Actual (Last 30 Days)"
        )

        fig, ax = plt.subplots()

        actual_30 = y_true[-30:]

        pred_30 = pred_on_hist[-30:]

        min_len = min(
            len(actual_30),
            len(pred_30)
        )

        ax.plot(
            actual_30[:min_len],
            label="Actual"
        )

        ax.plot(
            pred_30[:min_len],
            '--',
            label="Predicted"
        )

        ax.legend()

        st.pyplot(fig)

        # ====================================================
        # Gold vs Silver
        # ====================================================
        st.subheader(
            "🪙 Gold vs Silver Price Comparison"
        )

        fig, ax1 = plt.subplots(
            figsize=(10, 5)
        )

        ax1.plot(
            df["Date"],
            df["Gold_INR_g"],
            label="Gold"
        )

        ax1.set_ylabel(
            "Gold Price"
        )

        ax2 = ax1.twinx()

        ax2.plot(
            df["Date"],
            df["Silver_INR_g"],
            label="Silver"
        )

        ax2.set_ylabel(
            "Silver Price"
        )

        fig.tight_layout()

        st.pyplot(fig)

else:

    st.info(
        "⬆️ Please upload a CSV file to start prediction."
    )
