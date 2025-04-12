import dcc
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

import Reusable_dash_compotnets as drc
import figures as fig
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.MINTY, dbc.icons.FONT_AWESOME],)
# Load and prepare data

df = pd.read_csv("Flight_satisfaction.csv")
df = df.drop("id", axis=1)
df['satisfaction'] = df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)
df = pd.get_dummies(df, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'], drop_first=True)
df = df.dropna()
X = df.drop("satisfaction", axis=1)
y = df["satisfaction"]

"""
===========================================================================
Main Layout
"""


app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col([
                html.H2(

                    "Machin Learing  ",
                    className="text-center bg-primary text-white p-2",
                ),
                html.H4(
                    "Elias Leon",
                    className="text-center"
                ),
                html.H4(
                    "CS-150 : Community Action Computing",
                    className="text-center"
                ),
            ])
        ),
        dbc.Row(
            [
                dbc.Col( drc.card_component("Controls",[
                        drc.slider_component("slider-dataset-sample-size", "Sample Size", 1000, 25000, 5000, 10000),
                        drc.slider_component("slider-threshold", "Threshold", 0, 1, 0.001, 0.5)
                ]
                         ), width=12, lg=5, className="mt-4 border"),
                dbc.Col(
                    [
            dcc.Loading(dcc.Graph(id="graph-sklearn-svm")),
            dcc.Loading(dcc.Graph(id="graph-line-roc-curve")),
            html.Div(id="confusion-matrix-table", className="mt-4"),



                        html.Hr(),


                    ]
                ),
            ]
        ),
    ],
    fluid=True,
)
"""
==========================================================================
Callbacks
"""
@app.callback(
    Output("graph-sklearn-svm", "figure"),
    Output("graph-line-roc-curve", "figure"),
    Output("confusion-matrix-table", "children"),
    Input("slider-dataset-sample-size", "value"),
    Input("slider-threshold", "value"),
)
def update_graphs(sample_size, threshold):
    # Sample and split data
    X_sample = X.sample(n=sample_size, random_state=42)
    y_sample = y[X_sample.index]
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # For plotting: use PCA to reduce to 2D
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train_scaled)
    X_test_2d = pca.transform(X_test_scaled)

    # Create 2D mesh grid
    mesh_step = 0.2
    x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_step),
        np.arange(y_min, y_max, mesh_step)
    )

    # Generate predictions on the mesh (inverse transform to original space first)
    mesh_points_2d = np.c_[xx.ravel(), yy.ravel()]  # shape (N, 2)
    mesh_points_orig = pca.inverse_transform(mesh_points_2d)  # shape (N, n_features)
    Z = model.decision_function(mesh_points_orig)  # shape (N,)
    Z = Z.reshape(xx.shape)  # reshape for contour plot

    # Serve prediction figure
    prediction_figure = fig.serve_prediction_plot(
        model=model,
        X_train_orig=X_train_scaled,
        X_test_orig=X_test_scaled,
        X_train_2d=X_train_2d,
        X_test_2d= X_test_2d,
        y_train=y_train.values,
        y_test=y_test.values,
        Z=Z,
        xx=xx,
        yy=yy,
        mesh_step=mesh_step,
        threshold=threshold,
    )

    # ROC curve using true test features
    roc_figure = fig.serve_roc_curve(model, X_test_scaled, y_test)

    # Confusion matrix
    y_pred_test = (model.decision_function(X_test_scaled) > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()

    confusion_table = dbc.Table(
        children=[
            html.Thead(html.Tr([html.Th(), html.Th("Predicted Positive"), html.Th("Predicted Negative")])),
            html.Tbody([
                html.Tr([html.Th("Actual Positive"), html.Td(tp), html.Td(fn)]),
                html.Tr([html.Th("Actual Negative"), html.Td(fp), html.Td(tn)])
            ])
        ],
        bordered=True,
        color= "dark",
        hover=True,
        responsive=True,
        striped=True
    )

    return prediction_figure, roc_figure, confusion_table


if __name__ == "__main__":
    app.run(debug=True)
