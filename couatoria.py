import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_cytoscape as cyto
import pandas as pd
import networkx as nx
from collections import defaultdict
import base64
import io
import math

# Inicializar la aplicación Dash
app = dash.Dash(__name__)
server = app.server

# Diseño de la aplicación
app.layout = html.Div([
    html.H1("Análisis de Red de Co-autoría - Web of Science"),
    html.P("Carga tu archivo 'savedrecs.txt' de Web of Science para visualizar la red de co-autoría."),
    
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Arrastra y suelta o ', html.A('selecciona tu archivo')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),
    
    html.Div(id='output-data-upload'),
    
    html.Div([
        html.H3("Opciones de visualización"),
        html.Label("Umbral mínimo de colaboraciones:"),
        dcc.Slider(
            id='threshold-slider',
            min=1,
            max=10,
            step=1,
            value=1,
            marks={i: str(i) for i in range(1, 11)}
        ),
        html.Label("Layout de la red:"),
        dcc.Dropdown(
            id='layout-dropdown',
            options=[
                {'label': 'Force-Directed', 'value': 'cose'},
                {'label': 'Circular', 'value': 'circle'},
                {'label': 'Grid', 'value': 'grid'},
                {'label': 'Random', 'value': 'random'},
                {'label': 'Concentric', 'value': 'concentric'}
            ],
            value='cose'
        ),
        html.Label("Tamaño de nodo basado en:"),
        dcc.Dropdown(
            id='node-size-dropdown',
            options=[
                {'label': 'Grado (conexiones)', 'value': 'degree'},
                {'label': 'Número de publicaciones', 'value': 'publications'},
                {'label': 'Tamaño fijo', 'value': 'fixed'}
            ],
            value='degree'
        )
    ], style={'margin': '20px'}),
    
    html.Div([
        cyto.Cytoscape(
            id='cytoscape-network',
            layout={'name': 'cose'},
            style={'width': '100%', 'height': '600px'},
            elements=[],
            stylesheet=[
                {
                    'selector': 'node',
                    'style': {
                        'content': 'data(label)',
                        'background-color': '#0074D9',
                        'text-valign': 'center',
                        'color': 'white',
                        'text-outline-width': 2,
                        'text-outline-color': '#0074D9',
                        'width': 'data(size)',
                        'height': 'data(size)'
                    }
                },
                {
                    'selector': 'edge',
                    'style': {
                        'width': 'data(weight)',
                        'line-color': '#FF4136',
                        'opacity': 0.8,
                        'curve-style': 'bezier'
                    }
                },
                {
                    'selector': ':selected',
                    'style': {
                        'background-color': '#FF4136',
                        'line-color': '#FF4136',
                        'target-arrow-color': '#FF4136',
                        'source-arrow-color': '#FF4136'
                    }
                }
            ]
        )
    ]),
    
    html.Div(id='network-stats', style={'margin': '20px'})
])

# Función para procesar el archivo WoS
def parse_wos_file(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        text = decoded.decode('utf-8')
        records = text.split('\nER\n')
        articles_authors = []
        
        for record in records:
            if not record.strip():
                continue
            authors = []
            lines = record.split('\n')
            collecting_authors = False
            for line in lines:
                if line.startswith('AU '):
                    authors.append(line[3:].strip())
                    collecting_authors = True
                elif collecting_authors and (line.startswith('   ') or line.startswith('\t')):
                    authors.append(line.strip())
                elif line and not line.startswith('AU ') and not line.startswith('AU\t') and not line.startswith('   ') and not line.startswith('\t'):
                    collecting_authors = False
            if authors:
                articles_authors.append(authors)
        return articles_authors
    except Exception as e:
        print(e)
        return []

# Función para construir la red de co-autoría
def build_coauthorship_network(articles_authors, min_collaborations=1):
    # Crear un grafo vacío
    G = nx.Graph()
    
    # Contar publicaciones por autor
    publication_count = defaultdict(int)
    
    # Contar colaboraciones
    collaboration_count = defaultdict(int)
    
    for authors in articles_authors:
        # Contar publicaciones para cada autor
        for author in authors:
            publication_count[author.strip()] += 1
            
        # Para cada par de autores en el mismo artículo
        for i in range(len(authors)):
            author1 = authors[i].strip()
            if author1 not in G:
                G.add_node(author1, publications=0)
                
            for j in range(i+1, len(authors)):
                author2 = authors[j].strip()
                if author2 not in G:
                    G.add_node(author2, publications=0)
                
                # Ordenar los nombres para evitar duplicados (A-B y B-A)
                pair = tuple(sorted([author1, author2]))
                collaboration_count[pair] += 1
    
    # Actualizar conteo de publicaciones
    for author, count in publication_count.items():
        if author in G:
            G.nodes[author]['publications'] = count
    
    # Añadir aristas con el peso de colaboración
    for (author1, author2), count in collaboration_count.items():
        if count >= min_collaborations:
            G.add_edge(author1, author2, weight=count)
    
    return G

# Función para convertir NetworkX a elementos Cytoscape
def nx_to_cytoscape(G, size_based_on='degree'):
    elements = []
    
    if not G.nodes():
        return elements
    
    # Calcular tamaños de nodo
    if size_based_on == 'degree':
        sizes = dict(G.degree())
        max_degree = max(sizes.values()) if sizes else 1
        if max_degree == 0:
            max_degree = 1  # Evitar división por cero
        for node, degree in sizes.items():
            sizes[node] = 20 + 30 * (degree / max_degree)
    elif size_based_on == 'publications':
        publications = {node: data.get('publications', 1) for node, data in G.nodes(data=True)}
        max_pubs = max(publications.values()) if publications else 1
        if max_pubs == 0:
            max_pubs = 1  # Evitar división por cero
        for node, pubs in publications.items():
            sizes[node] = 20 + 30 * (pubs / max_pubs)
    else:  # fixed size
        sizes = {node: 20 for node in G.nodes()}
    
    # Añadir nodos
    for node, data in G.nodes(data=True):
        elements.append({
            'data': {
                'id': node, 
                'label': node,
                'size': sizes[node],
                'publications': data.get('publications', 1)
            }
        })
    
    # Añadir aristas
    for edge in G.edges(data=True):
        source, target, data = edge
        # Aumentar el peso visual para hacer las aristas más visibles
        visual_weight = 2 + math.log(data.get('weight', 1)) * 2
        elements.append({
            'data': {
                'source': source, 
                'target': target, 
                'weight': visual_weight,
                'collaborations': data.get('weight', 1)
            }
        })
    
    return elements

# Callback para procesar el archivo subido
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('cytoscape-network', 'elements'),
     Output('network-stats', 'children')],
    [Input('upload-data', 'contents'),
     Input('threshold-slider', 'value'),
     Input('layout-dropdown', 'value'),
     Input('node-size-dropdown', 'value')],
    [State('upload-data', 'filename')]
)
def update_output(contents, threshold, layout, node_size_based, filename):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    elements = []
    stats_html = html.Div()
    
    if contents is None:
        return html.Div(), elements, stats_html
    
    # Solo procesar cuando se sube un archivo o se cambia el umbral
    if trigger_id in ['upload-data', 'threshold-slider', 'node-size-dropdown', None]:
        articles_authors = parse_wos_file(contents)
        
        if not articles_authors:
            return html.Div(['Error al procesar el archivo o ningún autor encontrado.']), elements, stats_html
        
        # Construir la red
        G = build_coauthorship_network(articles_authors, threshold)
        
        # Convertir a formato Cytoscape
        elements = nx_to_cytoscape(G, node_size_based)
        
        # Calcular estadísticas
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        if num_nodes > 0:
            # Encontrar el autor más colaborador
            degrees = dict(G.degree())
            if degrees:
                max_degree_author = max(degrees.items(), key=lambda x: x[1])
            else:
                max_degree_author = ("N/A", 0)
                
            # Encontrar el autor con más publicaciones
            publications = {node: data.get('publications', 0) for node, data in G.nodes(data=True)}
            if publications:
                max_pubs_author = max(publications.items(), key=lambda x: x[1])
            else:
                max_pubs_author = ("N/A", 0)
                
            # Encontrar la colaboración más fuerte
            edge_weights = [data.get('weight', 0) for _, _, data in G.edges(data=True)]
            if edge_weights:
                max_weight = max(edge_weights)
            else:
                max_weight = 0
                
            stats_html = html.Div([
                html.H4("Estadísticas de la Red"),
                html.P(f"Número de autores (nodos): {num_nodes}"),
                html.P(f"Número de colaboraciones (aristas): {num_edges}"),
                html.P(f"Autor con más colaboraciones: {max_degree_author[0]} ({max_degree_author[1]} conexiones)"),
                html.P(f"Autor con más publicaciones: {max_pubs_author[0]} ({max_pubs_author[1]} publicaciones)"),
                html.P(f"Máximo de colaboraciones entre un par: {max_weight}"),
                html.P(f"Umbral mínimo de colaboraciones: {threshold}")
            ])
        
        return html.Div([f'Archivo procesado: {filename}', html.Br(), f'Se encontraron {len(articles_authors)} artículos.']), elements, stats_html
    
    # Si solo cambió el layout, mantener los elementos pero actualizar el layout
    elif trigger_id == 'layout-dropdown':
        return dash.no_update, dash.no_update, dash.no_update
    
    return html.Div(), elements, stats_html

# Callback para actualizar el layout
@app.callback(
    Output('cytoscape-network', 'layout'),
    [Input('layout-dropdown', 'value')]
)
def update_layout(layout_value):
    return {'name': layout_value}

if __name__ == '__main__':
    app.run(debug=True)