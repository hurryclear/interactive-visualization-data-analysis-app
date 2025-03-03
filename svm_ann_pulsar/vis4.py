import dash
import json
import numpy as np
from dash import dcc, html
from joblib import load
from dash.dependencies import Input, Output
from svm_model import train_model, svm_evaluate_model, svm_vis_boundary, svm_grid_train_params, svm_params_evaluation, svm_accuracy_heatmap, SVM_SAVE_PATH_LNEAR, SVM_SAVE_PATH_POLY, SVM_SAVE_PATH_RBF, SVM_SAVE_PATH_SIGMOID, C_RANGE, GAMMA_RANGE, DEGREE_RANGE
from knn_model import  MODEL1_EVAL_PATH, MODEL1_HISTORY_PATH, MODEL1_PATH, MODEL2_EVAL_PATH, MODEL2_HISTORY_PATH, MODEL2_PATH
from helper_functions import calculate_accuracy, node_link_topology_with_neuron_weights, learning_curves_dff, confusion_matrix, pre_data, build_line_diagram, evaluation_metrics


data = pre_data(2) # pre_data(2) returns (X_train, X_test, y_train, y_test, X_train_pca, X_test_pca, pca), where pca is the PCA object with 2 components

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
        html.P(
            "Best result",
            style={'font-size': '25px', 'color': 'blue'}
        ),
        html.P(
            "SVM kernel: poly, c=100, degree=3",
            style={'font-size': '25px', 'color': 'blue'}
        ),
        html.P(
            "SVM kernel: rbf, c=100, gamma=0.01",
            style={'font-size': '25px', 'color': 'blue'}
        )
    ], style={
        'border': '2px solid black',  # Black border
        'padding': '20px',  # Padding inside the rectangle
        'margin': '20px auto',  # Center the div with margin
        'width': '80%',  # Width of the rectangle
        'text-align': 'center'  # Center the text
    }),

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
                        3: {"label": "10", "style": {"font-size": "18px"}},     # Font size for mark 3
                        4: {"label": "100", "style": {"font-size": "18px"}},    # Font size for mark 4
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
                    "Result: c=0.1, accurary=0.9769.",
                    style={'font-size': '25px', 'color': 'blue'}
                )
            ], style={
                'border': '2px solid black',  # Black border
                'padding': '20px',  # Padding inside the rectangle
                'margin': '20px auto',  # Center the div with margin
                'width': '80%',  # Width of the rectangle
                'text-align': 'center'  # Center the text
            }),
        ]),
        # Poly SVM
        html.Div([
            html.H2(
                "SVM Kernel: Poly",
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
                        3: {"label": "10", "style": {"font-size": "18px"}},     # Font size for mark 3
                        4: {"label": "100", "style": {"font-size": "18px"}},    # Font size for mark 4
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
                dcc.Graph(id='accuracy-heatmap-poly', style={'flex': '40%'})
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'max-width': '1500px', 'margin': 'auto'}),  # Graphs side by side
            html.Div([
                html.P(
                    "Result: c=100, degree=3, accurary=0.9808.",
                    style={'font-size': '25px', 'color': 'blue'}
                )
            ], style={
                'border': '2px solid black',  # Black border
                'padding': '20px',  # Padding inside the rectangle
                'margin': '20px auto',  # Center the div with margin
                'width': '80%',  # Width of the rectangle
                'text-align': 'center'  # Center the text
            }),
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
                    min=0, max=4,  # Logical range for even spacing
                    marks={
                        0: {"label": "0.01", "style": {"font-size": "18px"}},  # Font size for mark 0
                        1: {"label": "0.1", "style": {"font-size": "18px"}},   # Font size for mark 1
                        2: {"label": "1", "style": {"font-size": "18px"}},     # Font size for mark 2
                        3: {"label": "10", "style": {"font-size": "18px"}},     # Font size for mark 3
                        4: {"label": "100", "style": {"font-size": "18px"}},    # Font size for mark 4
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
                    min=0, max=4,  # Logical range for even spacing
                    marks={
                        0: {"label": "0.01", "style": {"font-size": "18px"}},  # Font size for mark 0
                        1: {"label": "0.1", "style": {"font-size": "18px"}},  # Font size for mark 0
                        2: {"label": "1", "style": {"font-size": "18px"}},   # Font size for mark 1
                        3: {"label": "10", "style": {"font-size": "18px"}},     # Font size for mark 2
                        4: {"label": "100", "style": {"font-size": "18px"}},     # Font size for mark 3
                    },
                    step=None,  # Restrict slider to only these values
                    value=1,  # Default value: 1
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
                dcc.Graph(id='accuracy-heatmap-rbf', style={'flex': '40%'})
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'max-width': '1500px', 'margin': 'auto'}),  # Graphs side by side
            html.Div([
                html.P(
                    "Result: c=100, gamma=0.01, accurary=0.9808.",
                    style={'font-size': '25px', 'color': 'blue'}
                )
            ], style={
                'border': '2px solid black',  # Black border
                'padding': '20px',  # Padding inside the rectangle
                'margin': '20px auto',  # Center the div with margin
                'width': '80%',  # Width of the rectangle
                'text-align': 'center'  # Center the text
            }),
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
                        3: {"label": "10", "style": {"font-size": "18px"}},     # Font size for mark 3
                        4: {"label": "100", "style": {"font-size": "18px"}},    # Font size for mark 4
                    },
                    step=None,  # Restrict slider to only these values
                    value=2,  # Default value: 1 (logical position 3)
                    id='c-slider-sigmoid'
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
                    min=0, max=4,  # Logical range for even spacing
                    marks={
                        0: {"label": "0.01", "style": {"font-size": "18px"}},  # Font size for mark 0
                        1: {"label": "0.1", "style": {"font-size": "18px"}},  # Font size for mark 0
                        2: {"label": "1", "style": {"font-size": "18px"}},   # Font size for mark 1
                        3: {"label": "10", "style": {"font-size": "18px"}},     # Font size for mark 2
                        4: {"label": "100", "style": {"font-size": "18px"}},     # Font size for mark 3
                    },
                    step=None,  # Restrict slider to only these values
                    value=1,  # Default value: 1
                    id='gamma-slider-sigmoid'
                )
            ], style={'margin-bottom': '2px', 'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'}),

            # Container for decision boundary and evaluation metrics
            html.Div([
                dcc.Graph(id='decision-boundary-sigmoid', style={'flex': '50%', 'margin-right': '1px'}),
                dcc.Graph(id='confusion-matrix-sigmoid', style={'flex': '50%'}), 
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'max-width': '1500px', 'margin': 'auto'}),  # Graphs side by side
            html.Div([
                dcc.Graph(id='evaluation-metrics-sigmoid', style={'flex': '50%'}), 
                dcc.Graph(id='accuracy-heatmap-sigmoid', style={'flex': '50%'})
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'max-width': '1500px', 'margin': 'auto'}),  # Graphs side by side
            html.Div([
                html.P(
                    "Result: c=1, gamma=0.01, accurary=0.9769.",
                    style={'font-size': '25px', 'color': 'blue'}
                )
            ], style={
                'border': '2px solid black',  # Black border
                'padding': '20px',  # Padding inside the rectangle
                'margin': '20px auto',  # Center the div with margin
                'width': '80%',  # Width of the rectangle
                'text-align': 'center'  # Center the text
            }),
        ]),
    ]),

    html.H1(
        "Neural Network Visualization",
        style={
            'text-align': 'center',
            'font-size': '45px',  # Set the font size for the label
            'font-weight': 'bold',  # Optional: Make it bold
            'color': '#333'  # Optional: Change the text color
            }
    ),
    html.Div([
        html.P(
            "Analysis: ",
            style={'font-size': '25px', 'color': 'blue'}),
        html.P(
            "Result: Model1 (without hidden layer) with Accurary=0.9808.",
            style={'font-size': '25px', 'color': 'blue'}
        ),
        html.P(
            "Result: Model2 (with 1 hidden layer 8 Neuron) with Accurary=0.9789.",
            style={'font-size': '25px', 'color': 'blue'}
        )
    ], style={
        'border': '2px solid black',  # Black border
        'padding': '20px',  # Padding inside the rectangle
        'margin': '20px auto',  # Center the div with margin
        'width': '80%',  # Width of the rectangle
        'text-align': 'center'  # Center the text
    }),
    html.Div([
        # Left column for DFF evaluation
        html.Div([
            html.H1("Model 1: 8 Input, 1 Output, Activation: Logistic"),
            html.H2("Evaluation Metrics"),
            dcc.Graph(id="evaluation-metrics-model1", style={"height": "500px"}),  # Adjust height to fit well in the column

            html.H2("Learning Curves"),
            dcc.Graph(id="learning-curves-model1", style={"height": "500px"}),

            html.H2("Confusion Matrix"),
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
            html.H1("Model 2: 8 Input, 1 Hidden Layer with 8 Neurons, 1 Output, Activation: ReLU"), 
            html.H2("Evaluation Metrics"),
            dcc.Graph(id="evaluation-metrics-model2", style={"height": "500px"}),  # Adjust height to fit well in the column

            html.H2("Learning Curves"),
            dcc.Graph(id="learning-curves-model2", style={"height": "500px"}),

            html.H2("Confusion Matrix"),
            dcc.Graph(id="confusion-matrix-model2", style={"width":"200", "height": "500px"}),

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
    c_choose = C_RANGE[int(c_position)]


    # Initialize storage for all evaluations
    evaluations = {
        'all_metrics': {},  # Stores metrics across all C values
        'conf_matrix': None,
        'current_decision_boundary': None,
        'current_metrics': None
    }

    models_and_params_sigmoid = load(SVM_SAVE_PATH_SIGMOID)
    evaluation_metrics_sigmoid = svm_params_evaluation(models_and_params_sigmoid, data)

    # Pre-calculate metrics for all C values
    for c in C_RANGE:
        # x_min, x_max, y_min, y_max, Z, svc = train_model(data, "linear", c)
        # accuracy, precision, recall, f1, conf_matrix = svm_evaluate_model(data, svc)

        match_models_and_params_sigmoid_tuple = [entry for entry in models_and_params_sigmoid if entry[1] == c]
        _, _, _, _, svc = match_models_and_params_sigmoid_tuple[0]
        match_evaluation_metrics_sigmoid_tuple = [entry for entry in evaluation_metrics_sigmoid if entry[0] == c]
        _, _, _, accuracy, precision, recall, f1, conf_matrix = match_evaluation_metrics_sigmoid_tuple[0]
        
        # Store metrics with C as float key
        evaluations['all_metrics'][c] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        # Store visualization for selected C
        if c == c_choose:
            evaluations['conf_matrix'] = confusion_matrix(conf_matrix)
            evaluations['current_decision_boundary'] = svm_vis_boundary(data, svc)
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
    Output('accuracy-heatmap-poly', 'figure')],
    [Input('c-slider-poly', 'value'),
    Input('degree-slider-poly', 'value')]
)
def update_plot(c_position, degree_position):

    # Map slider position to actual C values
    c_choose = C_RANGE[int(c_position)]
    degree_choose = DEGREE_RANGE[int(degree_position)]

    # Initialize storage for all evaluations
    evaluations = {
        'all_metrics': {},  # Stores metrics across all C values
        'conf_matrix': None,
        'current_decision_boundary': None,
        'current_metrics': None
    }

    # models_and_params_poly = svm_grid_train_params(data, 'poly', c_range, gamma_range, degree_range)
    models_and_params_poly = load(SVM_SAVE_PATH_POLY)
    evaluation_metrics_poly = svm_params_evaluation(models_and_params_poly, data)
    accuracy_heatmap_poly = svm_accuracy_heatmap(evaluation_metrics_poly, param_x="c", param_y="degree")

    # Find the matching model and parameters
    match_models_and_params_poly_tuple = [entry for entry in models_and_params_poly if entry[1] == c_choose and entry[3] == degree_choose]
    _, _, _, _, svc = match_models_and_params_poly_tuple[0]
    match_evaluation_metrics_poly_tuple = [entry for entry in evaluation_metrics_poly if entry[0] == c_choose and entry[2] == degree_choose]
    _, _, _, accuracy, precision, recall, f1, conf_matrix = match_evaluation_metrics_poly_tuple[0]

    evaluations['current_decision_boundary'] = svm_vis_boundary(data, svc)
    evaluations['conf_matrix'] = confusion_matrix(conf_matrix)
    evaluations['current_metrics'] = evaluation_metrics(accuracy, precision, recall, f1)


    return (
        evaluations['current_decision_boundary'],
        evaluations['current_metrics'],
        evaluations['conf_matrix'],
        accuracy_heatmap_poly
    )
    

# RBF SVM
@app.callback(
    [Output('decision-boundary-rbf', 'figure'),
    Output('evaluation-metrics-rbf', 'figure'),
    Output('confusion-matrix-rbf', 'figure'),
    Output('accuracy-heatmap-rbf', 'figure')],
    [Input('c-slider-rbf', 'value'),
    Input('gamma-slider-rbf', 'value')]
)

def update_plot(c_position, gamma_position):

    # Map slider position to actual C values
    c_choose = C_RANGE[int(c_position)]
    gamma_choose = GAMMA_RANGE[int(gamma_position)]

    # Initialize storage for all evaluations
    evaluations = {
        'all_metrics': {},  # Stores metrics across all C values
        'conf_matrix': None,
        'current_decision_boundary': None,
        'current_metrics': None
    }

    # models_and_params_poly = svm_grid_train_params(data, 'poly', c_range, gamma_range, degree_range)
    models_and_params_rbf = load(SVM_SAVE_PATH_RBF)
    evaluation_metrics_rbf = svm_params_evaluation(models_and_params_rbf, data)
    accuracy_heatmap_rbf = svm_accuracy_heatmap(evaluation_metrics_rbf, param_x="c", param_y="gamma")

    # Find the matching model and parameters from the trained models
    match_models_and_params_rbf_tuple = [entry for entry in models_and_params_rbf if entry[1] == c_choose and entry[2] == gamma_choose]
    _, _, _, _, svc = match_models_and_params_rbf_tuple[0]
    match_evaluation_metrics_rbf_tuple = [entry for entry in evaluation_metrics_rbf if entry[0] == c_choose and entry[1] == gamma_choose]
    _, _, _, accuracy, precision, recall, f1, conf_matrix = match_evaluation_metrics_rbf_tuple[0]

    evaluations['current_decision_boundary'] = svm_vis_boundary(data, svc)
    evaluations['conf_matrix'] = confusion_matrix(conf_matrix)
    evaluations['current_metrics'] = evaluation_metrics(accuracy, precision, recall, f1)

    return (
        evaluations['current_decision_boundary'],
        evaluations['current_metrics'],
        evaluations['conf_matrix'],
        accuracy_heatmap_rbf
    )


# Sigmoid SVM
@app.callback(
    [Output('decision-boundary-sigmoid', 'figure'),
    Output('evaluation-metrics-sigmoid', 'figure'),
    Output('confusion-matrix-sigmoid', 'figure'),
    Output('accuracy-heatmap-sigmoid', 'figure')],
    [Input('c-slider-sigmoid', 'value'),
    Input('gamma-slider-sigmoid', 'value')]
)

def update_plot(c_position, gamma_position):

    # Map slider position to actual C values
    c_choose = C_RANGE[int(c_position)]
    gamma_choose = GAMMA_RANGE[int(gamma_position)]

    # Initialize storage for all evaluations
    evaluations = {
        'all_metrics': {},  # Stores metrics across all C values
        'conf_matrix': None,
        'current_decision_boundary': None,
        'current_metrics': None
    }

    # models_and_params_poly = svm_grid_train_params(data, 'poly', c_range, gamma_range, degree_range)
    models_and_params_sigmoid = load(SVM_SAVE_PATH_SIGMOID)
    evaluation_metrics_sigmoid = svm_params_evaluation(models_and_params_sigmoid, data)
    accuracy_heatmap_sigmoid = svm_accuracy_heatmap(evaluation_metrics_sigmoid, param_x="c", param_y="gamma")

    match_models_and_params_sigmoid_tuple = [entry for entry in models_and_params_sigmoid if entry[1] == c_choose and entry[2] == gamma_choose]
    # Find the matching model and parameters from the trained models
    _, _, _, _, svc = match_models_and_params_sigmoid_tuple[0]
    match_evaluation_metrics_sigmoid_tuple = [entry for entry in evaluation_metrics_sigmoid if entry[0] == c_choose and entry[1] == gamma_choose]
    _, _, _, accuracy, precision, recall, f1, conf_matrix = match_evaluation_metrics_sigmoid_tuple[0]

    evaluations['current_decision_boundary'] = svm_vis_boundary(data, svc)
    evaluations['conf_matrix'] = confusion_matrix(conf_matrix)
    evaluations['current_metrics'] = evaluation_metrics(accuracy, precision, recall, f1)

    return (
        evaluations['current_decision_boundary'],
        evaluations['current_metrics'],
        evaluations['conf_matrix'],
        accuracy_heatmap_sigmoid
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
    confusion_matrix_fig_dff = confusion_matrix(conf_matrix)
    

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
    confusion_matrix_fig_dff = confusion_matrix(conf_matrix)
    

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