from textwrap import dedent

from dash import dcc, html
import dash_bootstrap_components as dbc

def slider_component(slider_id, label, min_val=0, max_val=100, step=1, value=5):
    return html.Div([
        html.Label(label),
        dcc.Slider(
            id=slider_id,
            min=min_val,
            max=max_val,
            step=step,
            value=value,
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], className="mb-4")

def card_component(title, children):
    return dbc.Card([
        dbc.CardHeader(title),
        dbc.CardBody(children)
    ])

