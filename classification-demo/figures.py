import colorlover as cl
import plotly.graph_objs as go
import numpy as np
from dash import dash_table
from sklearn import metrics



def serve_prediction_plot(
    model,
    X_train_orig,
    X_test_orig,  # Full 24D data for model computations
    X_train_2d,
    X_test_2d,       # 2D PCA-transformed data for plotting
    y_train,
    y_test,
    Z, xx, yy, mesh_step, threshold
):
    # Predictions using the full-dimension data
    y_pred_train = (model.decision_function(X_train_orig) > threshold).astype(int)
    y_pred_test = (model.decision_function(X_test_orig) > threshold).astype(int)
    train_score = metrics.accuracy_score(y_true=y_train, y_pred=y_pred_train)
    test_score = metrics.accuracy_score(y_true=y_test, y_pred=y_pred_test)

    # Compute threshold for decision contour
    scaled_threshold = threshold * (Z.max() - Z.min()) + Z.min()
    value_range = max(abs(scaled_threshold - Z.min()), abs(scaled_threshold - Z.max()))

    # Colorscale
    bright_cscale = [[0, "#ff3700"], [1, "#0b8bff"]]
    cscale = [
        [0.0000000, "#ff744c"],
        [0.1428571, "#ff916d"],
        [0.2857143, "#ffc0a8"],
        [0.4285714, "#ffe7dc"],
        [0.5714286, "#e5fcff"],
        [0.7142857, "#c8feff"],
        [0.8571429, "#9af8ff"],
        [1.0000000, "#20e6ff"],
    ]

    # Contour and decision boundary
    trace0 = go.Contour(
        x=np.arange(xx.min(), xx.max(), mesh_step),
        y=np.arange(yy.min(), yy.max(), mesh_step),
        z=Z.reshape(xx.shape),
        zmin=scaled_threshold - value_range,
        zmax=scaled_threshold + value_range,
        hoverinfo="none",
        showscale=False,
        contours=dict(showlines=False),
        colorscale=cscale,
        opacity=0.9,
    )

    trace1 = go.Contour(
        x=np.arange(xx.min(), xx.max(), mesh_step),
        y=np.arange(yy.min(), yy.max(), mesh_step),
        z=Z.reshape(xx.shape),
        showscale=False,
        hoverinfo="none",
        contours=dict(
            showlines=False, type="constraint", operation="=", value=scaled_threshold
        ),
        name=f"Threshold ({scaled_threshold:.3f})",
        line=dict(color="#708090"),
    )

    trace2 = go.Scatter(
        x=X_train_2d[:, 0],
        y=X_train_2d[:, 1],
        mode="markers",
        name=f"Training Data (accuracy={train_score:.3f})",
        marker=dict(size=10, color=y_train, colorscale=bright_cscale),
    )

    trace3 = go.Scatter(
        x=X_test_2d[:, 0],
        y=X_test_2d[:, 1],
        mode="markers",
        name=f"Test Data (accuracy={test_score:.3f})",
        marker=dict(size=10, symbol="triangle-up", color=y_test, colorscale=bright_cscale),
    )

    layout = go.Layout(
        xaxis=dict(ticks="", showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(ticks="", showticklabels=False, showgrid=False, zeroline=False),
        hovermode="closest",
        legend=dict(x=0, y=-0.01, orientation="h"),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    return go.Figure(data=[trace0, trace1, trace2, trace3], layout=layout)


def serve_roc_curve(model, X_test, y_test):
    decision_test = model.decision_function(X_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, decision_test)

    # AUC Score
    auc_score = metrics.roc_auc_score(y_true=y_test, y_score=decision_test)

    trace0 = go.Scatter(
        x=fpr, y=tpr, mode="lines", name="Test Data", marker={"color": "#13c6e9"}
    )

    layout = go.Layout(
        title=f"ROC Curve (AUC = {auc_score:.3f})",
        xaxis=dict(title="False Positive Rate", gridcolor="#2f3445"),
        yaxis=dict(title="True Positive Rate", gridcolor="#2f3445"),
        legend=dict(x=0, y=1.05, orientation="h"),
        margin=dict(l=100, r=10, t=25, b=40),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    data = [trace0]
    figure = go.Figure(data=data, layout=layout)

    return figure


def confusion_matrix(model, X_test, y_test, Z, threshold):
    # Compute scaled threshold
    scaled_threshold = threshold * (Z.max() - Z.min()) + Z.min()

    # Predict based on threshold
    y_pred_test = (model.decision_function(X_test) > scaled_threshold).astype(int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_test).ravel()

    # Format as 2x2 matrix with labels
    table_data = [
        {"Actual \\ Predicted": "Negative", "Negative": tn, "Positive": fp},
        {"Actual \\ Predicted": "Positive", "Negative": fn, "Positive": tp}
    ]

    # Create the Dash table
    table = dash_table.DataTable(
        id="confusion-matrix-2x2",
        columns=[
            {"name": "Actual \\ Predicted", "id": "Actual \\ Predicted"},
            {"name": "Negative", "id": "Negative"},
            {"name": "Positive", "id": "Positive"}
        ],
        data=table_data,
        style_table={"width": "60%", "margin": "20px auto"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
        style_cell={"textAlign": "center", "padding": "10px"},
    )

    return table