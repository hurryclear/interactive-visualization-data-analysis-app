import os
import dash
import json
import base64
import numpy as np
from dash import dcc, html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from svm_model import train_model, evaluate_model, visua_decision_boundary, evaluation_metrics
from knn_model import  MODEL1_EVAL_PATH, MODEL1_HISTORY_PATH, MODEL1_PATH, MODEL2_EVAL_PATH, MODEL2_HISTORY_PATH, MODEL2_PATH
from helper_functions import calculate_accuracy, node_link_topology_with_neuron_weights, learning_curves_dff, confusion_matrix_dff, pre_data, build_line_diagram, convert_image_to_base64

# MODEL1_BLOCK_TOPOLOGY_PATH = "./model1/dff_model_topology.png"
# MODEL2_BLOCK_TOPOLOGY_PATH = "./model2/dff_model_topology.png"

data = pre_data(2)

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([

    html.H1(
        "SVM Decision Boundary Visualization",
        style={
            'text-align': 'center',
            'font-size': '40px',  # Set the font size for the label
            'font-weight': 'bold',  # Optional: Make it bold
            'color': '#333'  # Optional: Change the text color
            }
    ),

    html.Div([
        # linear SVM
        html.Div([
            html.H2(
                "SVM Kernel: Linear",
                style={
                        'font-size': '30px',  # Set the font size for the label
                        'font-weight': 'bold',  # Optional: Make it bold
                        'color': '#333'  # Optional: Change the text color
                    }
            ),
            # Slider for parameter 'C'
            html.Div([
                html.Label(
                    "Adjust Regularization Parameter C:",
                    style={
                        'font-size': '20px',  # Set the font size for the label
                        'font-weight': 'bold',  # Optional: Make it bold
                        'color': '#333'  # Optional: Change the text color
                    }
                ),
                dcc.Slider(
                    min=0,
                    max=4,
                    marks={
                        0: {"label": "0.01", "style": {"font-size": "18px"}},  # Font size for mark 0
                        1: {"label": "0.1", "style": {"font-size": "18px"}},   # Font size for mark 1
                        2: {"label": "1", "style": {"font-size": "18px"}},     # Font size for mark 2
                        3: {"label": "5", "style": {"font-size": "18px"}},     # Font size for mark 3
                        4: {"label": "10", "style": {"font-size": "18px"}},    # Font size for mark 4
                    },
                    step=None,  # Restrict slider to only these values
                    value=1,  # Default value
                    id='c-slider-linear'
                )
            ], style={
                'margin-bottom': '2px',
                'width': '80%',
                'margin-left': 'auto',
                'margin-right': 'auto'
            }),  # Add spacing below the slider

            # Container for decision boundary and evaluation metrics
            html.Div([
                dcc.Graph(id='decision-boundary-linear', style={'flex': '50%', 'margin-right': '1px'}),
                dcc.Graph(id='confusion-matrix-linear', style={'flex': '50%'}), 
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'max-width': '1500px', 'margin': 'auto'}),  # Graphs side by side
            html.Div([
                dcc.Graph(id='evaluation-metrics-linear', style={'flex': '50%'}), 
                dcc.Graph(id='line-diagram-linear', style={'flex': '50%'})
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'max-width': '1500px', 'margin': 'auto'}),  # Graphs side by side
            html.Div([
                html.P(
                    "Analysis: When we raise the c value, we can see increase of the evaluation values, but after 0.1 there is no big difference, so we would choose c=0.1 as best parameter, where accurary=0.9792.",
                    style={'font-size': '25px', 'color': 'blue'}
                ),
                html.P(
                    "Result: c=0.1, accurary=0.9792.",
                    style={'font-size': '25px', 'color': 'blue'}
                )
            ]),
        ]),
        # Poly SVM
        html.Div([
            html.H2(
                "SVM Kernel: Ploy",
                style={
                        'font-size': '30px',  # Set the font size for the label
                        'font-weight': 'bold',  # Optional: Make it bold
                        'color': '#333'  # Optional: Change the text color
                    }
            ),
            # Slider for parameter 'C'
            html.Div([
                html.Label(
                    "Adjust Regularization Parameter C:", 
                    style={
                        'font-size': '20px',  # Set the font size for the label
                        'font-weight': 'bold',  # Optional: Make it bold
                        'color': '#333'  # Optional: Change the text color
                    }),
                dcc.Slider(
                    min=0, max=4,  # Logical range for even spacing
                    marks={
                        0: {"label": "0.01", "style": {"font-size": "18px"}},  # Font size for mark 0
                        1: {"label": "0.1", "style": {"font-size": "18px"}},   # Font size for mark 1
                        2: {"label": "1", "style": {"font-size": "18px"}},     # Font size for mark 2
                        3: {"label": "5", "style": {"font-size": "18px"}},     # Font size for mark 3
                        4: {"label": "10", "style": {"font-size": "18px"}},    # Font size for mark 4
                    },
                    step=None,  # Restrict slider to only these values
                    value=4,  # Default value: 1 (logical position 3)
                    id='c-slider-poly'
                )
            ], style={'margin-bottom': '2px', 'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'}),  # Add spacing below the slider

            html.Div([
                html.Label(
                    "Adjust Degree:",
                    style={
                        'font-size': '20px',  # Set the font size for the label
                        'font-weight': 'bold',  # Optional: Make it bold
                        'color': '#333'  # Optional: Change the text color
                    }
                ),
                dcc.Slider(
                    min=0, max=8,  # Logical range for even spacing
                    marks={
                        0: {"label": "2", "style": {"font-size": "18px"}},  # Font size for mark 0
                        1: {"label": "3", "style": {"font-size": "18px"}},   # Font size for mark 1
                        2: {"label": "4", "style": {"font-size": "18px"}},     # Font size for mark 2
                        3: {"label": "5", "style": {"font-size": "18px"}},     # Font size for mark 3
                        4: {"label": "6", "style": {"font-size": "18px"}},    # Font size for mark 4
                        5: {"label": "7", "style": {"font-size": "18px"}},  # Font size for mark 0
                        6: {"label": "8", "style": {"font-size": "18px"}},   # Font size for mark 1
                        7: {"label": "9", "style": {"font-size": "18px"}},     # Font size for mark 2
                        8: {"label": "10", "style": {"font-size": "18px"}},     # Font size for mark 3
                    },
                    step=None,  # Restrict slider to only these values
                    value=1,  # Default value: 3
                    id='degree-slider-poly'
                )
            ], style={'margin-bottom': '2px', 'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'}),

            # Container for decision boundary and evaluation metrics
            html.Div([
                dcc.Graph(id='decision-boundary-poly', style={'flex': '50%', 'margin-right': '1px'}),
                dcc.Graph(id='confusion-matrix-poly', style={'flex': '50%'}), 
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'max-width': '1500px', 'margin': 'auto'}),  # Graphs side by side
            html.Div([
                dcc.Graph(id='evaluation-metrics-poly', style={'flex': '50%'}), 
                dcc.Graph(id='line-diagram-poly', style={'flex': '50%'})
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'max-width': '1500px', 'margin': 'auto'}),  # Graphs side by side
            html.Div([
                html.P(
                    "Analysis: When we raise the c value, the evaluation values increase (although the accuracy no big difference, but others change greatly), base on that we choose the c value as 10 and when we fix c vlaue and change the degree, we can find the best degree is 3, where accurary=0.9800.",
                    style={'font-size': '25px', 'color': 'blue'}
                ),
                html.P(
                    "Result: c=10, degree=3, accurary=0.9800.",
                    style={'font-size': '25px', 'color': 'blue'}
                )
            ]),
        ]),
        # RBF SVM
        html.Div([
            html.H2(
                "SVM Kernel: RBF",
                style={
                        'font-size': '30px',  # Set the font size for the label
                        'font-weight': 'bold',  # Optional: Make it bold
                        'color': '#333'  # Optional: Change the text color
                    }
            ),
            # Slider for parameter 'C'
            html.Div([
                html.Label(
                    "Adjust Regularization Parameter C:",
                    style={
                        'font-size': '20px',  # Set the font size for the label
                        'font-weight': 'bold',  # Optional: Make it bold
                        'color': '#333'  # Optional: Change the text color
                    }
                ),
                dcc.Slider(
                    min=0, max=5,  # Logical range for even spacing
                    marks={
                        0: {"label": "0.01", "style": {"font-size": "18px"}},  # Font size for mark 0
                        1: {"label": "0.1", "style": {"font-size": "18px"}},   # Font size for mark 1
                        2: {"label": "1", "style": {"font-size": "18px"}},     # Font size for mark 2
                        3: {"label": "5", "style": {"font-size": "18px"}},     # Font size for mark 3
                        4: {"label": "10", "style": {"font-size": "18px"}},    # Font size for mark 4
                    },
                    step=None,  # Restrict slider to only these values
                    value=3,  # Default value: 1 (logical position 3)
                    id='c-slider-rbf'
                )
            ], style={'margin-bottom': '2px', 'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'}),  # Add spacing below the slider
            html.Div([
                html.Label(
                    "Adjust Gamma:",
                    style={
                        'font-size': '20px',  # Set the font size for the label
                        'font-weight': 'bold',  # Optional: Make it bold
                        'color': '#333'  # Optional: Change the text color
                    }
                ),
                dcc.Slider(
                    min=0, max=3,  # Logical range for even spacing
                    marks={
                        0: {"label": "0.1", "style": {"font-size": "18px"}},  # Font size for mark 0
                        1: {"label": "1", "style": {"font-size": "18px"}},   # Font size for mark 1
                        2: {"label": "5", "style": {"font-size": "18px"}},     # Font size for mark 2
                        3: {"label": "10", "style": {"font-size": "18px"}},     # Font size for mark 3
                        4: {"label": "20", "style": {"font-size": "18px"}},    # Font size for mark 4
                    },
                    step=None,  # Restrict slider to only these values
                    value=2,  # Default value: 1
                    id='gamma-slider-rbf'
                )
            ], style={'margin-bottom': '2px', 'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'}),

            # Container for decision boundary and evaluation metrics
            html.Div([
                dcc.Graph(id='decision-boundary-rbf', style={'flex': '50%', 'margin-right': '1px'}),
                dcc.Graph(id='confusion-matrix-rbf', style={'flex': '50%'}), 
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'max-width': '1500px', 'margin': 'auto'}),  # Graphs side by side
            html.Div([
                dcc.Graph(id='evaluation-metrics-rbf', style={'flex': '50%'}), 
                dcc.Graph(id='line-diagram-rbf', style={'flex': '50%'})
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'max-width': '1500px', 'margin': 'auto'}),  # Graphs side by side
            html.Div([
                html.P(
                    "Analysis: We can find when the c value is 5 we have best evaluation, except precision, but that’s influence is small, so we take c = 5. We fix c = 5 and change gamma and can find the best value is 5. In this case (c=5, gamma=5), we have accuracy=0.9497.",
                    style={'font-size': '25px', 'color': 'blue'}
                ),
                html.P(
                    "Result: c=5, gamma=5, accurary=0.9497.",
                    style={'font-size': '25px', 'color': 'blue'}
                )
            ]),
        ]),
        # Sigmoid SVM
        html.Div([
            html.H2(
                "SVM Kernel: Sigmoid",
                style={
                        'font-size': '30px',  # Set the font size for the label
                        'font-weight': 'bold',  # Optional: Make it bold
                        'color': '#333'  # Optional: Change the text color
                    }
            ),
            # Slider for parameter 'C'
            html.Div([
                html.Label(
                    "Adjust Regularization Parameter C:",
                    style={
                        'font-size': '20px',  # Set the font size for the label
                        'font-weight': 'bold',  # Optional: Make it bold
                        'color': '#333'  # Optional: Change the text color
                    }
                ),
                dcc.Slider(
                    min=0, max=4,  # Logical range for even spacing
                    marks={
                        0: {"label": "0.01", "style": {"font-size": "18px"}},  # Font size for mark 0
                        1: {"label": "0.1", "style": {"font-size": "18px"}},   # Font size for mark 1
                        2: {"label": "1", "style": {"font-size": "18px"}},     # Font size for mark 2
                        3: {"label": "5", "style": {"font-size": "18px"}},     # Font size for mark 3
                        4: {"label": "10", "style": {"font-size": "18px"}},    # Font size for mark 4
                    },
                    step=None,  # Restrict slider to only these values
                    value=0,  # Default value: 1 (logical position 3)
                    id='c-slider-sigmoid'
                )
            ], style={'margin-bottom': '2px', 'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'}),  # Add spacing below the slider

            # Container for decision boundary and evaluation metrics
            html.Div([
                dcc.Graph(id='decision-boundary-sigmoid', style={'flex': '50%', 'margin-right': '1px'}),
                dcc.Graph(id='confusion-matrix-sigmoid', style={'flex': '50%'}), 
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'max-width': '1500px', 'margin': 'auto'}),  # Graphs side by side
            html.Div([
                dcc.Graph(id='evaluation-metrics-sigmoid', style={'flex': '50%'}), 
                dcc.Graph(id='line-diagram-sigmoid', style={'flex': '50%'})
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'max-width': '1500px', 'margin': 'auto'}),  # Graphs side by side
            html.Div([
                html.P(
                    "Analysis: When we raise the c value, we can see decrease of accuracy and other values are always too low, we ignore their influnce for choosing c value, so we would choose c=0.01 as best parameter, where accurary=0.8591.",
                    style={'font-size': '25px', 'color': 'blue'}
                ),
                html.P(
                    "Result: c=0.01, accurary=0.8591.",
                    style={'font-size': '25px', 'color': 'blue'}
                )
            ]),
        ]),
    ]),

    html.H1(
        "Deep Feedforward Neural Network Visualization",
        style={
            'text-align': 'center',
            'font-size': '45px',  # Set the font size for the label
            'font-weight': 'bold',  # Optional: Make it bold
            'color': '#333'  # Optional: Change the text color
            }
    ),
    html.Div([
        html.P(
            "We choose model 2 to be the better one, which has higher accuracy "
            "and also other evaluation values are higher. ",
            style={
                'font-size': '25px',
                'color': 'blue'
            }
        ),
        html.P(
            "Although the model 2 has only 1 hidden layer with 8 neurons, "
            "considering our dataset, it is enough to get very good results.",
            style={
                'font-size': '25px',
                'color': 'blue'
            }
        )
    ]),

    html.Div([
        # Left column for DFF evaluation
        html.Div([
            html.H1("Model 1: 8 Input, 1 Hidden Layer with 2 Neurons, 1 Output"),
            html.H2("DFF Evaluation Metrics"),
            dcc.Graph(id="evaluation-metrics-model1", style={"height": "500px"}),  # Adjust height to fit well in the column

            html.H2("DFF Learning Curves"),
            dcc.Graph(id="learning-curves-model1", style={"height": "500px"}),

            html.H2("DFF Confusion Matrix"),
            dcc.Graph(id="confusion-matrix-model1", style={"width":"200", "height": "500px"}),

            # Topology
        #     html.H2("DFF Topology"),
        #     html.Img(id="block-topology-model1", style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
        ], style={
            "width": "48%",       # Occupies 48% of the width
            "display": "inline-block",  # Side-by-side layout
            "vertical-align": "top",    # Aligns content to the top
            "padding-right": "10px"     # Adds spacing to the right
        }),
        # Right column for ... evaluation
        html.Div([
            html.H1("Model 2: 8 Input, 1 Hidden Layer with 8 Neurons, 1 Output"), 
            html.H2("DFF Evaluation Metrics"),
            dcc.Graph(id="evaluation-metrics-model2", style={"height": "500px"}),  # Adjust height to fit well in the column

            html.H2("DFF Learning Curves"),
            dcc.Graph(id="learning-curves-model2", style={"height": "500px"}),

            html.H2("DFF Confusion Matrix"),
            dcc.Graph(id="confusion-matrix-model2", style={"width":"200", "height": "500px"}),

            # Topology
        #     html.H2("DFF Topology"),
        #     html.Img(id="block-topology-model2", style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
        ], style={
            "width": "48%",       # Occupies 48% of the width
            "display": "inline-block",  # Side-by-side layout
            "vertical-align": "top",    # Aligns content to the top
            "padding-left": "10px"     # Adds spacing to the right
        }),

        html.H2("Node Link Topology 1"),
        dcc.Graph(id="node-topology-model1"),

        html.H2("Node Link Topology 2"),
        dcc.Graph(id="node-topology-model2"),
    ]),
])

# Linear SVM
@app.callback(
    [Output('decision-boundary-linear', 'figure'),
    Output('evaluation-metrics-linear', 'figure'),
    Output('confusion-matrix-linear', 'figure'),
    Output('line-diagram-linear', 'figure')],
    [Input('c-slider-linear', 'value')]
)
def update_plot(c_position):
    # Map slider position to actual C values
    c_values = [0.01, 0.1, 1, 5, 10]
    c_choose = c_values[int(c_position)]

    # Initialize storage for all evaluations
    evaluations = {
        'all_metrics': {},  # Stores metrics across all C values
        'conf_matrix': None,
        'current_decision_boundary': None,
        'current_metrics': None
    }

    # Pre-calculate metrics for all C values
    for c in c_values:
        x_min, x_max, y_min, y_max, Z, svc = train_model(data, "linear", c)
        accuracy, precision, recall, f1, conf_matrix = evaluate_model(data, svc)
        
        # Store metrics with C as float key
        evaluations['all_metrics'][c] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        # Store visualization for selected C
        if c == c_choose:
            evaluations['conf_matrix'] = confusion_matrix_dff(conf_matrix)
            evaluations['current_decision_boundary'] = visua_decision_boundary(data, x_min, x_max, y_min, y_max, Z)
            evaluations['current_metrics'] = evaluation_metrics(accuracy, precision, recall, f1)

    # Build the line diagram using all metrics
    line_diagram_fig = build_line_diagram(evaluations['all_metrics'])

    return (
        evaluations['current_decision_boundary'],
        evaluations['current_metrics'],
        evaluations['conf_matrix'],
        line_diagram_fig
    )


# Poly SVM
@app.callback(
    [Output('decision-boundary-poly', 'figure'),
    Output('evaluation-metrics-poly', 'figure'),
    Output('confusion-matrix-poly', 'figure'),
    Output('line-diagram-poly', 'figure')],
    [Input('c-slider-poly', 'value'),
    Input('degree-slider-poly', 'value')]
)
def update_plot(c_position, degree_position):

    # Map slider position to actual C values
    c_values = [0.01, 0.1, 1, 5, 10]
    c_choose = c_values[int(c_position)]
    degree_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # Define degree options
    degree = degree_values[int(degree_position)]

    # Initialize storage for all evaluations
    evaluations = {
        'all_metrics': {},  # Stores metrics across all C values
        'conf_matrix': None,
        'current_decision_boundary': None,
        'current_metrics': None
    }

    # Pre-calculate metrics for all C values
    for c in c_values:
        x_min, x_max, y_min, y_max, Z, svc = train_model(data, "poly", c, degree=degree) 
        accuracy, precision, recall, f1, conf_matrix = evaluate_model(data, svc)
        
        # Store metrics with C as float key
        evaluations['all_metrics'][c] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        # Store visualization for selected C
        if c == c_choose:
            evaluations['conf_matrix'] = confusion_matrix_dff(conf_matrix)
            evaluations['current_decision_boundary'] = visua_decision_boundary(data, x_min, x_max, y_min, y_max, Z)
            evaluations['current_metrics'] = evaluation_metrics(accuracy, precision, recall, f1)

    # Build the line diagram using all metrics
    line_diagram_fig = build_line_diagram(evaluations['all_metrics'])

    return (
        evaluations['current_decision_boundary'],
        evaluations['current_metrics'],
        evaluations['conf_matrix'],
        line_diagram_fig
    )
    

# RBF SVM
@app.callback(
    [Output('decision-boundary-rbf', 'figure'),
    Output('evaluation-metrics-rbf', 'figure'),
    Output('confusion-matrix-rbf', 'figure'),
    Output('line-diagram-rbf', 'figure')],
    [Input('c-slider-rbf', 'value'),
    Input('gamma-slider-rbf', 'value')]
)
def update_plot(c_position, gamma_position):

    # x_min, x_max, y_min, y_max, Z, svc = train_model(data, "rbf", c, gamma=gamma) # gamma should be changable
    # accuracy, precision, recall, f1, conf_matrix = evaluate_model(data, svc)
    # decision_boundary_rbf = visua_decision_boundary(data, x_min, x_max, y_min, y_max, Z)
    # evaluation_metrics_rbf = evaluation_metrics(accuracy, precision, recall, f1)
    # return decision_boundary_rbf, evaluation_metrics_rbf


    # Map slider position to actual C values
    c_values = [0.01, 0.1, 1, 5, 10]
    c_choose = c_values[int(c_position)]
    gamma_values = [0.1, 1, 5, 10]
    gamma = gamma_values[int(gamma_position)]

    # Initialize storage for all evaluations
    evaluations = {
        'all_metrics': {},  # Stores metrics across all C values
        'conf_matrix': None,
        'current_decision_boundary': None,
        'current_metrics': None
    }

    # Pre-calculate metrics for all C values
    for c in c_values:
        x_min, x_max, y_min, y_max, Z, svc = train_model(data, "rbf", c, gamma=gamma) # 
        accuracy, precision, recall, f1, conf_matrix = evaluate_model(data, svc)
        
        # Store metrics with C as float key
        evaluations['all_metrics'][c] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        # Store visualization for selected C
        if c == c_choose:
            evaluations['conf_matrix'] = confusion_matrix_dff(conf_matrix)
            evaluations['current_decision_boundary'] = visua_decision_boundary(data, x_min, x_max, y_min, y_max, Z)
            evaluations['current_metrics'] = evaluation_metrics(accuracy, precision, recall, f1)

    # Build the line diagram using all metrics
    line_diagram_fig = build_line_diagram(evaluations['all_metrics'])

    return (
        evaluations['current_decision_boundary'],
        evaluations['current_metrics'],
        evaluations['conf_matrix'],
        line_diagram_fig
    )
    

# Sigmoid SVM
@app.callback(
    [Output('decision-boundary-sigmoid', 'figure'),
    Output('evaluation-metrics-sigmoid', 'figure'),
    Output('confusion-matrix-sigmoid', 'figure'),
    Output('line-diagram-sigmoid', 'figure')],
    [Input('c-slider-sigmoid', 'value')]
)
def update_plot(c_position):

    c_values = [0.01, 0.1, 1, 5, 10]
    c_choose = c_values[int(c_position)]

    # Initialize storage for all evaluations
    evaluations = {
        'all_metrics': {},  # Stores metrics across all C values
        'conf_matrix': None,
        'current_decision_boundary': None,
        'current_metrics': None
    }

    # Pre-calculate metrics for all C values
    for c in c_values:
        x_min, x_max, y_min, y_max, Z, svc = train_model(data, "sigmoid", c, gamma=2, degree=3)
        accuracy, precision, recall, f1, conf_matrix = evaluate_model(data, svc)
        
        # Store metrics with C as float key
        evaluations['all_metrics'][c] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        # Store visualization for selected C
        if c == c_choose:
            evaluations['conf_matrix'] = confusion_matrix_dff(conf_matrix)
            evaluations['current_decision_boundary'] = visua_decision_boundary(data, x_min, x_max, y_min, y_max, Z)
            evaluations['current_metrics'] = evaluation_metrics(accuracy, precision, recall, f1)

    # Build the line diagram using all metrics
    line_diagram_fig = build_line_diagram(evaluations['all_metrics'])

    return (
        evaluations['current_decision_boundary'],
        evaluations['current_metrics'],
        evaluations['conf_matrix'],
        line_diagram_fig
    )

# Model1
@app.callback(
    [Output("evaluation-metrics-model1", "figure"),
    Output("learning-curves-model1", "figure"),
    Output("confusion-matrix-model1", "figure")],
    [Input("learning-curves-model1", "id")]  # A dummy input to trigger the callback once
)
def update_graphs(_):

    # Load training history
    with open(MODEL1_HISTORY_PATH, 'r') as f:
        history = json.load(f)
    # Load evaluation metrics
    with open(MODEL1_EVAL_PATH, 'r') as f:
        evaluation = json.load(f)
    conf_matrix = np.array(evaluation["confusion_matrix"])
    classification_report = evaluation["classification_report"]

    # Calculate accuracy from the confusion matrix
    accuracy = calculate_accuracy(conf_matrix)

    # Extract weighted averages for precision, recall, and F1-score
    precision = classification_report["weighted avg"]["precision"]
    recall = classification_report["weighted avg"]["recall"]
    f1 = classification_report["weighted avg"]["f1-score"]

    # Generate visualizations
    evaluation_metrics_dff = evaluation_metrics(accuracy, precision, recall, f1)
    learning_curves_fig_dff = learning_curves_dff(history)
    confusion_matrix_fig_dff = confusion_matrix_dff(conf_matrix)
    

    return  evaluation_metrics_dff, learning_curves_fig_dff, confusion_matrix_fig_dff

@app.callback(
    Output("node-topology-model1", "figure"),
    [Input("learning-curves-model1", "id")]  # A dummy input to trigger the callback once
)
def update_graphs(_):
    
    # Generate and encode topology diagram
    # topology_image_path = block_topology(MODEL1_PATH, MODEL1_BLOCK_TOPOLOGY_PATH)
    # with open(topology_image_path, "rb") as img_file:
    #     block_topology_dff = "data:image/png;base64," + base64.b64encode(img_file.read()).decode()

    node_link_topology_fig = node_link_topology_with_neuron_weights(MODEL1_PATH)

    return  node_link_topology_fig

# Model2
@app.callback(
    [Output("evaluation-metrics-model2", "figure"),
    Output("learning-curves-model2", "figure"),
    Output("confusion-matrix-model2", "figure")],
    [Input("learning-curves-model2", "id")]  # A dummy input to trigger the callback once
)
def update_graphs(_):

    # Load training history
    with open(MODEL2_HISTORY_PATH, 'r') as f:
        history = json.load(f)
    # Load evaluation metrics
    with open(MODEL2_EVAL_PATH, 'r') as f:
        evaluation = json.load(f)
    conf_matrix = np.array(evaluation["confusion_matrix"])
    classification_report = evaluation["classification_report"]

    # Calculate accuracy from the confusion matrix
    accuracy = calculate_accuracy(conf_matrix)

    # Extract weighted averages for precision, recall, and F1-score
    precision = classification_report["weighted avg"]["precision"]
    recall = classification_report["weighted avg"]["recall"]
    f1 = classification_report["weighted avg"]["f1-score"]

    # Generate visualizations
    evaluation_metrics_dff = evaluation_metrics(accuracy, precision, recall, f1)
    learning_curves_fig_dff = learning_curves_dff(history)
    confusion_matrix_fig_dff = confusion_matrix_dff(conf_matrix)
    

    return  evaluation_metrics_dff, learning_curves_fig_dff, confusion_matrix_fig_dff

@app.callback(
    Output("node-topology-model2", "figure"),
    [Input("learning-curves-model2", "id")]  # A dummy input to trigger the callback once
)
def update_graphs(_):
    
    # Generate and encode topology diagram
    # topology_image_path = block_topology(MODEL2_PATH, MODEL2_BLOCK_TOPOLOGY_PATH)
    # with open(topology_image_path, "rb") as img_file:
    #     block_topology_dff = "data:image/png;base64," + base64.b64encode(img_file.read()).decode()

    node_link_topology_fig = node_link_topology_with_neuron_weights(MODEL2_PATH)

    return  node_link_topology_fig



# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)