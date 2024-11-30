#%pip install regex
import pandas as pd
import regex as re
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, silhouette_score, davies_bouldin_score, silhouette_samples,mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dash import Dash, dcc, html, Input, Output
from skimage import measure

df = pd.read_csv('wein.csv')
df = df.dropna()
df = df.drop_duplicates()
# convert all columns to strings to handle leading zeros and punctuation
df = df.applymap(str)
# drop the punct at the end of the string/number
df = df.applymap(lambda x: x.rstrip('.') if isinstance(x, str) else x)
# drop the 0 in the beginning of the string/number
df = df.applymap(lambda x: re.sub(r'^0+(?=\d)','',x) if isinstance(x, str) else x)
# convert columns back to appropriate numeric types
df = df.apply(pd.to_numeric, errors='coerce')
# print(df.dtypes)
# df
#df.to_csv('wein_clean.csv', index=False)
df.head()

# %pip install scikit-learn dash plotly scikit-image

data = df
# Standardization scales the data so that all features contribute equally to the clustering process, especially when features have different ranges.
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
# Regressionen berechnen
def calculate_regressions(data):
    '''
    for each pair of columns in the data, calculate the R² of a linear regression
    param data: a pandas DataFrame
    return: a pandas DataFrame with columns 'X', 'Y', 'R²'
    '''
    dic_list = [
        {'X': x, 'Y': y, 'R^2': r2_score(data[y], LinearRegression().fit(data[[x]], data[y]).predict(data[[x]]))}
        for x, y in combinations(data.columns, 2)
    ]
    return pd.DataFrame(dic_list)

regression_results = calculate_regressions(data)

# Dash App 
app = Dash(__name__)
app.layout = html.Div([
    # 1. Linear Regression
    html.Div([
        html.H2("Lineare Regression", style={'textAlign': 'center'}),  
        # Dropdown für x und y Achsen
    html.Div([
        html.Div([
            html.Label("x-Achse"),
            dcc.Dropdown(id='x_variable',value='Alle_Phenole', placeholder="Wähle X", style={'width': '100%'})
        ], style={'width': '45%', 'display': 'inline-block'}),

        html.Div([
            html.Label("y-Achse"),
            dcc.Dropdown(id='y_variable',value='Flavanoide', style={'width': '100%'})
        ], style={'width': '45%', 'display': 'inline-block', 'marginLeft': '5%'})
    ]),

        dcc.Graph(id='regression-plot'),
        html.Div(id='regression-stats')
    ]),

    # 2. KMeans Clustering
    html.Div([
        html.H2("KMeans Clustering", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='kmeans-dropdown',
            options=[
                {'label': 'K=2', 'value': 2},
                {'label': 'K=3', 'value': 3},
                {'label': 'K=4', 'value': 4},
                {'label': 'K=5', 'value': 5},
            ],
            value=2,
        ),
        # 2.1 Graph for visualizing the KMeans Clustering
        dcc.Graph(id='cluster-graph'),
        # 2.2 Silhouette plot for each data point
        dcc.Graph(id='silhouette-graph'),
        # 2.3 Graph for evaluating metrics
        dcc.Graph(id='evaluation-graph'),
        # 2.4 Scree plot for PCA explained variance ratio
        dcc.Graph(id='scree-plot'),
    ])

])
# 1. Linear Regression
@app.callback(
    [Output('x_variable', 'options'),
    Output('y_variable', 'options'),
    Output('regression-plot', 'figure'),
    Output('regression-stats', 'children')],
    Input('x_variable', 'value'),
    Input('y_variable', 'value')
    
)
def update_content(x, y):
    x_columns = [{'label': col, 'value': col} for col in data.columns if col !=y]
    y_columns = [{'label': col, 'value': col} for col in data.columns if col != x]

    if not x or not y:
        return x_columns, y_columns, {}, "Bitte X und Y auswählen."

    X, y_vals = data[[x]].values, data[y].values
    model = LinearRegression().fit(X, y_vals)

    r2 = r2_score(y_vals, model.predict(X))
    mse = mean_squared_error(y_vals, model.predict(X))
    rmse = mse ** 0.5  # Quadratwurzel von MSE
    mae = mean_absolute_error(y_vals, model.predict(X))
    slope, intercept = model.coef_[0], model.intercept_
    fig = px.scatter(x=X.flatten(), y=y_vals, labels={'x': x, 'y': y})
    fig.add_scatter(x=X.flatten(), y=model.predict(X), mode='lines', name='Regressionslinie')

    # Statistiken
    stats = html.Div([
        html.P(f"R²: {r2:.3f}"),
        html.P(f"MSE (Mean Squared Error): {mse:.3f}"),
        html.P(f"RMSE (Root Mean Squared Error): {rmse:.3f}"),
        html.P(f"MAE (Mean Absolute Error): {mae:.3f}"),
        html.P(f"Achsenabschnitt: {intercept:.3f}"),
        html.P(f"Steigung: {slope:.3f}"),
    ],  style={
    'padding': '10px',  # Innenabstand
    #'border': '1px solid black',
    'borderRadius': '8px',  # Abgerundete Ecken
    'boxShadow': '2px 2px 10px rgba(0, 0, 0, 0.1)',
    'width': '50%', 
    'margin': '20px auto',#zentrieren
    'fontSize': '14px',
    'backgroundColor': '#f9f9f9'  
})
    return x_columns, y_columns, fig, stats

# 2. KMeans Clustering
@app.callback(
    [Output('cluster-graph', 'figure'), 
    Output('silhouette-graph', 'figure'),
    Output('evaluation-graph', 'figure'),
    Output('scree-plot', 'figure')],
    [Input('kmeans-dropdown', 'value')]
)
def update_clustering(n_clusters):

    # functions for clustering, it returns
    def perform_kmeans(data, n_clusters):

        # KMeans Algorithm (k-means++ variant): The algorithm initializes the cluster centers in a smart way to speed up convergence. 
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++')

        # Cluster Labels: Assigns each data point to a cluster
        labels = kmeans.fit_predict(data)

        # return labels and kmeans object, notice that kmeans is a object that contains the cluster centers, and other information
        return labels, kmeans

    def evaluate_clustering(data, labels):
        # Silhouette Score: The silhouette score measures how similar an object is to its own cluster compared to other clusters. It returns Mean Silhouette Coefficient for all samples. By defaul using euclidean distance.
        silhouette_avg = silhouette_score(data, labels) # the higher the better
        # Davies-Bouldin Index: The Davies-Bouldin index is defined as the average similarity measure of each cluster with its most similar cluster. It returns the average index of all clusters.
        davies_bouldin_avg = davies_bouldin_score(data, labels) # the lower the better
        return silhouette_avg, davies_bouldin_avg

    # 2.0: Data Preprocessing
    # PCA for visualization: For high-dimensional data, we reduce the data to 2D using PCA for visualization.
    pca = PCA(n_components=2) # return 2 principal components
    data_pca = pca.fit_transform(data_scaled)
    # Perform clustering with selected number of clusters
    labels, kmeans = perform_kmeans(data_scaled, n_clusters)
    # Evaluate metrics for the chosen number of clusters
    silhouette_avg, davies_bouldin_avg = evaluate_clustering(data_scaled, labels)
    # center of the clusters in the PCA space
    cluster_centers_pca = pca.transform(kmeans.cluster_centers_)

    # 2.1: Scatter plot with decision boundaries
    # decision boundaries (voronoi-like)
    x_min, x_max = data_pca[:, 0].min() - 1, data_pca[:, 0].max() + 1
    y_min, y_max = data_pca[:, 1].min() - 1, data_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_labels = kmeans.predict(pca.inverse_transform(grid_points)).reshape(xx.shape)

    # Scatter plot with decision boundaries
    cluster_fig = go.Figure()
    # Define a shared colormap for clusters
    cluster_colors = ['rgba(147, 181, 198, 0.9)',  # Blue
                'rgba(221, 237, 170, 0.9)',  # Yellow
                'rgba(240, 207, 101, 0.9)',  # Red
                'rgba(215, 129, 106, 0.9)',  # Green
                'rgba(189, 79, 108, 0.9)']  # Green
    # Add decision boundaries
    for cluster in np.unique(labels):
        cluster_area = grid_labels == cluster
        cluster_fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, 500),
            y=np.linspace(y_min, y_max, 500),
            z=cluster_area.astype(int),
            colorscale=[
                [0, 'rgba(255,255,255,0)'], 
                [1, cluster_colors[cluster]]
            ],
            opacity=0.6,
            showscale=False,
            name=f'Cluster {cluster + 1}',
            hoverinfo='skip',
        ))
        cluster_fig.add_trace(go.Scatter(
            x=[None], y=[None], 
            mode='markers',
            marker=dict(
                size=10,
                color=cluster_colors[cluster],
            ),
            name=f'Cluster {cluster + 1}',
        ))
    # Add data points (set color to black)
    for cluster in np.unique(labels):
        cluster_fig.add_trace(go.Scatter(
            x=data_pca[labels == cluster, 0],
            y=data_pca[labels == cluster, 1],
            mode='markers',
            name=f'Cluster {cluster + 1}',
            marker=dict(size=6, color='black'),  # Set color to black
            showlegend=False,
        ))
    
    # Add cluster centers
    cluster_centers_pca = pca.transform(kmeans.cluster_centers_)
    for i, (x, y) in enumerate(cluster_centers_pca):
        cluster_fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(color='red', size=12, symbol='star'),
            name=f'Center {i + 1}',
            showlegend=False,          
        )
    )
    # Update layout
    cluster_fig.update_layout(
        title=f'K-Means Clustering with {n_clusters} Clusters',
        xaxis_title='PCA1',
        yaxis_title='PCA2',
        legend_title='Clusters',
        height=600,
        legend=dict(
            x=1.05,
            y=1,
            bordercolor='black',
            borderwidth=1
        )
    )


    # 2.2: Silhouette plot for each data point
    def create_silhouette_plot(data_scaled, labels, n_clusters):
        # Compute silhouette average
        silhouette_avg = silhouette_score(data_scaled, labels)
        
        # Compute silhouette values for each sample
        silhouette_values = silhouette_samples(data_scaled, labels)
        y_lower = 10  # Start position for first cluster
        silhouette_fig = go.Figure()

        for k in range(n_clusters):
            # Filter and sort silhouette values for cluster k
            ith_cluster_silhouette_values = silhouette_values[np.array(labels) == k]
            ith_cluster_silhouette_values.sort()

            # Determine cluster size and range for plotting
            size_cluster_k = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_k

            # Add bars for silhouette values
            silhouette_fig.add_trace(go.Bar(
                x=ith_cluster_silhouette_values,
                y=np.arange(y_lower, y_upper),
                orientation='h',
                name=f'Cluster {k+1}',
                marker_color=cluster_colors[k],
            ))

            # Annotate cluster number
            silhouette_fig.add_annotation(
                x=-0.05,  # Position annotation to the left
                y=y_lower + 0.5 * size_cluster_k,
                text=str(k+1),
                showarrow=False,
                font=dict(size=10, color='black')
            )

            # Update position for next cluster
            y_lower = y_upper + 10

        # Add a vertical line for the average silhouette score
        silhouette_fig.add_shape(
            type='line',
            x0=silhouette_avg,
            x1=silhouette_avg,
            y0=0,
            y1=y_lower,
            line=dict(color='red', dash='dash')
        )

        # Annotate average silhouette score
        silhouette_fig.add_annotation(
            x=silhouette_avg,
            y=y_lower + 10,  # Position annotation above the plot
            text=f"Silhouette Avg: {silhouette_avg:.4f}",
            showarrow=False,
            font=dict(size=12, color="red")
        )

        # Update layout for the plot
        silhouette_fig.update_layout(
            title="Silhouette Plot for Various Clusters",
            xaxis_title="Silhouette Coefficient",
            yaxis_title="Cluster Label",
            yaxis=dict(showticklabels=False, showgrid=False, title_font=dict(size=12)),
            xaxis=dict(
                tickvals=[-0.1, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Specify the tick positions
                ticktext=["-0.1", "0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],  # Custom tick labels
                range=[-0.1, 1.0]  # Set the visible range for the x-axis
            ),
            margin=dict(l=50, r=50, t=50, b=50),
            height=600,
            legend_title="Cluster",
            plot_bgcolor="rgba(240, 240, 240, 0.5)",
        )

        return silhouette_fig
    # silhouette_fig = go.Figure()
    silhouette_fig = create_silhouette_plot(data_scaled, labels, n_clusters)


    # 2.3: Evaluate metrics for all cluster configurations
    cluster_range = range(2, 6)
    silhouette_scores = [] # list to store silhouette scores
    davies_bouldin_scores = [] # list to store davies bouldin scores
    
    for k in cluster_range:
        labels, _ = perform_kmeans(data_scaled, k)
        silhouette_avg, davies_bouldin_avg = evaluate_clustering(data_scaled, labels)
        silhouette_scores.append(silhouette_avg)
        davies_bouldin_scores.append(davies_bouldin_avg)

    # Create evaluation metrics plot
    evaluation_fig = go.Figure()
    evaluation_fig.add_trace(go.Scatter(
        x=list(cluster_range), y=silhouette_scores,
        mode='lines+markers', name='Silhouette Coefficient'
    ))
    evaluation_fig.add_trace(go.Scatter(
        x=list(cluster_range), y=davies_bouldin_scores,
        mode='lines+markers', name='Davies-Bouldin Index'
    ))
    evaluation_fig.update_layout(
        title='Evaluation Metrics for K-Means Clustering',
        xaxis_title='Number of Clusters',
        yaxis_title='Score',
        legend_title='Metrics'
    )


    # 2.4: Create scree plot
    pca_full = PCA()
    pca_full.fit(data_scaled)
    explained_variance = pca_full.explained_variance_ratio_
    
    scree_fig = go.Figure()
    scree_fig.add_trace(
        go.Scatter(
            x=list(range(1, len(explained_variance) + 1)), 
            y=explained_variance, 
            mode='lines+markers', 
            name='Explained Variance'
        )
    )
    scree_fig.update_layout(
        title='Scree Plot', 
        xaxis_title='Principal Component', 
        yaxis_title='Explained Variance Ratio'
    )
    
    return cluster_fig, silhouette_fig, evaluation_fig, scree_fig

if __name__ == '__main__':
    app.run_server(debug=True)



