import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Feature Selection Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('feature_selection_all_classifiers_results.csv')
    return df

df = load_data()

st.title("🔬 Feature Selection & Model Performance Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 Model Comparison",
    "🎯 Feature Impact",
    "🏆 Best Models",
    "🔍 Custom Analysis"
])

with tab1:
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Experiments", len(df))
    with col2:
        st.metric("Unique Models", df['Model'].nunique())
    with col3:
        st.metric("Feature Selection Methods", df['Feature_Selection'].nunique())
    with col4:
        st.metric("Max Features", df['N_Features'].max())

    st.subheader("Feature Selection Methods")
    fs_counts = df['Feature_Selection'].value_counts()
    fig = px.bar(x=fs_counts.index, y=fs_counts.values,
                 labels={'x': 'Feature Selection Method', 'y': 'Count'},
                 title='Experiments by Feature Selection Method')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Statistical Summary")
    metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'PR_AUC']
    st.dataframe(df[metrics_cols].describe(), use_container_width=True)

with tab2:
    st.header("Model Comparison Across Metrics")

    metric_choice = st.selectbox(
        "Select Metric",
        ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'PR_AUC']
    )

    fs_method = st.selectbox(
        "Select Feature Selection Method",
        ['All'] + list(df['Feature_Selection'].unique())
    )

    filtered_df = df if fs_method == 'All' else df[df['Feature_Selection'] == fs_method]

    st.subheader(f"{metric_choice} by Model and Feature Selection")
    avg_by_model = filtered_df.groupby(['Model', 'Feature_Selection'])[metric_choice].mean().reset_index()
    fig = px.bar(avg_by_model, x='Model', y=metric_choice, color='Feature_Selection',
                 title=f'Average {metric_choice} by Model',
                 height=600, barmode='group')
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Multiple Metrics Comparison")
    selected_models = st.multiselect(
        "Select Models to Compare",
        df['Model'].unique(),
        default=list(df['Model'].unique())[:5]
    )

    if selected_models:
        model_df = filtered_df[filtered_df['Model'].isin(selected_models)]

        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'PR_AUC']

        fig = go.Figure()
        for model in selected_models:
            model_data = model_df[model_df['Model'] == model]
            avg_metrics = [model_data[m].mean() for m in metrics_to_plot]
            fig.add_trace(go.Bar(
                name=model,
                x=metrics_to_plot,
                y=avg_metrics
            ))

        fig.update_layout(
            barmode='group',
            showlegend=True,
            title="Average Metrics Comparison by Model",
            xaxis_title="Metrics",
            yaxis_title="Score",
            height=600,
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Feature Impact Analysis")

    st.subheader("Performance vs Number of Features")

    selected_metric = st.selectbox(
        "Select Performance Metric",
        ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'PR_AUC'],
        key='feature_metric'
    )

    models_for_feature = st.multiselect(
        "Select Models",
        df['Model'].unique(),
        default=list(df['Model'].unique())[:8],
        key='models_feature'
    )

    if models_for_feature:
        feature_df = df[df['Model'].isin(models_for_feature)]

        fig = px.line(feature_df, x='N_Features', y=selected_metric,
                      color='Model', markers=True,
                      title=f'{selected_metric} vs Number of Features by Model',
                      height=600)
        st.plotly_chart(fig, use_container_width=True)

        avg_by_fs_features = feature_df.groupby(['Feature_Selection', 'N_Features'])[selected_metric].mean().reset_index()
        fig2 = px.line(avg_by_fs_features, x='N_Features', y=selected_metric,
                      color='Feature_Selection', markers=True,
                      title=f'{selected_metric} vs Features (by Feature Selection Method)',
                      height=600)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Feature Selection Method Effectiveness")

    avg_by_fs = df.groupby('Feature_Selection')[['Accuracy', 'F1_Score', 'ROC_AUC']].mean().reset_index()

    fig = go.Figure()
    for metric in ['Accuracy', 'F1_Score', 'ROC_AUC']:
        fig.add_trace(go.Bar(
            name=metric,
            x=avg_by_fs['Feature_Selection'],
            y=avg_by_fs[metric]
        ))

    fig.update_layout(
        barmode='group',
        title='Average Performance by Feature Selection Method',
        xaxis_title='Feature Selection Method',
        yaxis_title='Score',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Best Models Analysis")

    st.subheader("Top Models by Single Metric")

    col1, col2 = st.columns(2)
    with col1:
        ranking_metric = st.selectbox(
            "Rank by Metric",
            ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'PR_AUC'],
            key='ranking'
        )
    with col2:
        top_n = st.slider("Number of Top Models", 5, 50, 10)

    top_models = df.nlargest(top_n, ranking_metric)[['Model', 'Feature_Selection', 'N_Features',
                                                       'Accuracy', 'Precision', 'Recall',
                                                       'F1_Score', 'ROC_AUC', 'PR_AUC']]
    st.dataframe(top_models.reset_index(drop=True), use_container_width=True)

    fig = px.bar(top_models, x='Model', y=ranking_metric, color='Feature_Selection',
                 hover_data=['N_Features'],
                 title=f'Top {top_n} Models by {ranking_metric}',
                 height=500)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Custom Analysis & Filtering")

    st.subheader("Filter Data")

    col1, col2 = st.columns(2)
    with col1:
        selected_fs_methods = st.multiselect(
            "Feature Selection Methods",
            df['Feature_Selection'].unique(),
            default=list(df['Feature_Selection'].unique())
        )
    with col2:
        selected_models_custom = st.multiselect(
            "Models",
            df['Model'].unique(),
            default=list(df['Model'].unique())
        )

    min_features = st.slider("Minimum Number of Features",
                             int(df['N_Features'].min()),
                             int(df['N_Features'].max()),
                             int(df['N_Features'].min()))

    filtered_custom = df[
        (df['Feature_Selection'].isin(selected_fs_methods)) &
        (df['Model'].isin(selected_models_custom)) &
        (df['N_Features'] >= min_features)
    ]

    st.write(f"Filtered Results: {len(filtered_custom)} rows")

    st.subheader("Custom Visualization")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        plot_type = st.selectbox("Plot Type", ['Line', 'Bar'])
    with col2:
        x_axis = st.selectbox("X-Axis",
                             ['N_Features', 'Model', 'Feature_Selection'])
    with col3:
        y_axis = st.selectbox("Y-Axis",
                             ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'PR_AUC',
                              'Train_Time_s', 'Predict_Time_s'])
    with col4:
        color_by = st.selectbox("Color By", ['Model', 'Feature_Selection'])

    if len(filtered_custom) > 0:
        if x_axis == 'N_Features' and plot_type == 'Line':
            avg_custom = filtered_custom.groupby([x_axis, color_by])[y_axis].mean().reset_index()
            fig = px.line(avg_custom, x=x_axis, y=y_axis,
                         color=color_by, markers=True,
                         title=f'Average {y_axis} vs {x_axis}',
                         height=600)
        else:
            avg_custom = filtered_custom.groupby([x_axis, color_by])[y_axis].mean().reset_index()
            fig = px.bar(avg_custom, x=x_axis, y=y_axis,
                        color=color_by, barmode='group',
                        title=f'Average {y_axis} by {x_axis}',
                        height=600)
            fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Filtered Data Table")
        st.dataframe(filtered_custom, use_container_width=True)

        csv = filtered_custom.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_results.csv",
            mime="text/csv"
        )
    else:
        st.warning("No data matches the current filters")

st.sidebar.title("About")
st.sidebar.info("""
This dashboard visualizes feature selection and model performance results.

**Features:**
- Overview statistics
- Model comparison across metrics
- Feature impact analysis
- Best model identification
- Performance & efficiency analysis
- Custom filtering and visualization

**Metrics:**
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- PR AUC
""")

st.sidebar.title("Quick Stats")
st.sidebar.metric("Best Accuracy", f"{df['Accuracy'].max():.4f}")
st.sidebar.metric("Best F1 Score", f"{df['F1_Score'].max():.4f}")
st.sidebar.metric("Best ROC AUC", f"{df['ROC_AUC'].max():.4f}")
best_model = df.loc[df['F1_Score'].idxmax(), 'Model']
st.sidebar.info(f"Best Model (F1): {best_model}")
