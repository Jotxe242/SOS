import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_cytoscape as cyto
import pandas as pd
import networkx as nx
from collections import defaultdict
import base64
import io
import math
import ast
import igraph as ig
import leidenalg
import random

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
        ),
        html.Label("Algoritmo de detección de comunidades:"),
        dcc.Dropdown(
            id='community-algorithm-dropdown',
            options=[
                {'label': 'Leiden', 'value': 'leiden'}
            ],
            value='leiden'
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
                        'background-color': 'data(color)',  # Use node color
                        'text-valign': 'center',
                        'color': 'white',
                        'text-outline-width': 2,
                        'text-outline-color': '#222',
                        'width': 'data(size)',
                        'height': 'data(size)'
                    }
                },
                {
                    'selector': 'edge',
                    'style': {
                        'width': 'data(weight)',
                        'line-color': 'data(color)',  # Use edge color
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
    
    html.Div(id='network-stats', style={'margin': '20px'}),
    
    html.Div(id='pair-to-articles', style={'display': 'none'}),  # <-- Add this line

    html.Div(id='edge-info', style={'margin': '20px'})
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

def get_community_colors(num_communities):
    # Generate distinct colors for each community (using HSL for variety)
    colors = []
    for i in range(num_communities):
        hue = int(360 * i / max(1, num_communities))
        colors.append(f"hsl({hue}, 70%, 50%)")
    return colors

def nx_to_cytoscape(G, size_based_on='degree', community_algorithm='leiden'):
    elements = []
    if not G.nodes():
        return elements, 0

    # Compute communities
    if community_algorithm == 'leiden':
        communities = compute_leiden_communities(G)
        num_communities = len(set(communities.values()))
    else:
        communities = {node: -1 for node in G.nodes()}
        num_communities = 0

    # Assign colors to communities
    community_colors = get_community_colors(num_communities)
    node_colors = {node: community_colors[communities[node]] if communities[node] >= 0 else "#888" for node in G.nodes()}

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
                'publications': data.get('publications', 1),
                'community': communities.get(node, -1),
                'color': node_colors[node]  # Add color attribute
            }
        })
    
    # Añadir aristas
    for edge in G.edges(data=True):
        source, target, data = edge
        visual_weight = 2 + math.log(data.get('weight', 1)) * 4
        # Edge color: use source node's community color
        edge_color = node_colors[source]
        elements.append({
            'data': {
                'source': source, 
                'target': target, 
                'weight': visual_weight,
                'collaborations': data.get('weight', 1),
                'color': edge_color
            }
        })
    
    return elements, num_communities

def compute_leiden_communities(G):
    # Convert NetworkX graph to igraph with node names
    nx_g = nx.Graph(G)  # Ensure undirected
    node_names = list(nx_g.nodes())
    edges = [(node_names.index(u), node_names.index(v)) for u, v in nx_g.edges()]
    ig_g = ig.Graph(edges=edges, directed=False)
    ig_g.add_vertices(len(node_names) - ig_g.vcount())  # Ensure all nodes are present
    ig_g.vs["name"] = node_names  # Set node names

    # Run Leiden algorithm
    partition = leidenalg.find_partition(ig_g, leidenalg.ModularityVertexPartition)
    membership = partition.membership
    communities = {node_names[idx]: comm for idx, comm in enumerate(membership)}
    return communities

# Función para procesar el archivo WoS
def parse_wos_institutions(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        text = decoded.decode('utf-8')
        records = text.split('\nER\n')
        articles_institutions = []
        for record in records:
            if not record.strip():
                continue
            institutions = set()
            lines = record.split('\n')
            for line in lines:
                if line.startswith('C3 '):
                    # Remove 'C3 ' and split by ';'
                    c3_content = line[3:].strip()
                    for inst in c3_content.split(';'):
                        inst_clean = inst.strip()
                        if inst_clean:
                            institutions.add(inst_clean)
            if institutions:
                articles_institutions.append(list(institutions))
        return articles_institutions
    except Exception as e:
        print(e)
        return []

def build_institution_network(articles_institutions, min_collaborations=1):
    G = nx.Graph()
    collaboration_count = defaultdict(int)
    for institutions in articles_institutions:
        for i in range(len(institutions)):
            inst1 = institutions[i]
            if inst1 not in G:
                G.add_node(inst1)
            for j in range(i+1, len(institutions)):
                inst2 = institutions[j]
                if inst2 not in G:
                    G.add_node(inst2)
                pair = tuple(sorted([inst1, inst2]))
                collaboration_count[pair] += 1
    for (inst1, inst2), count in collaboration_count.items():
        if count >= min_collaborations:
            G.add_edge(inst1, inst2, weight=count)
    return G

def parse_wos_references(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        text = decoded.decode('utf-8')
        records = text.split('\nER\n')
        articles_references = []
        for record in records:
            if not record.strip():
                continue
            references = set()
            lines = record.split('\n')
            for line in lines:
                if line.startswith('CR '):
                    cr_content = line[3:].strip()
                    if cr_content:
                        references.add(cr_content)
                elif line.startswith('   ') or line.startswith('\t'):
                    cr_content = line.strip()
                    if cr_content:
                        references.add(cr_content)
            articles_references.append(references)
        return articles_references
    except Exception as e:
        print(e)
        return []

def build_bibliographic_coupling_network(articles_references, min_shared_refs=1):
    G = nx.Graph()
    num_papers = len(articles_references)
    # Add nodes for each paper
    for idx in range(num_papers):
        G.add_node(f"Paper {idx+1}", references=len(articles_references[idx]))
    # Compare each pair of papers
    for i in range(num_papers):
        for j in range(i+1, num_papers):
            shared_refs = articles_references[i] & articles_references[j]
            weight = len(shared_refs)
            if weight >= min_shared_refs:
                G.add_edge(f"Paper {i+1}", f"Paper {j+1}", weight=weight)
    return G

# Callback para procesar el archivo subido
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('cytoscape-network', 'elements'),
     Output('network-stats', 'children')],
    [Input('upload-data', 'contents'),
     Input('threshold-slider', 'value'),
     Input('layout-dropdown', 'value'),
     Input('node-size-dropdown', 'value'),
     Input('community-algorithm-dropdown', 'value')],
    [State('upload-data', 'filename')]
)
def update_output(contents, threshold, layout, node_size_based, community_algorithm, filename):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    elements = []
    stats_html = html.Div()

    if contents is None:
        return html.Div(), elements, stats_html

    # Parse references and build bibliographic coupling network
    articles_references = parse_wos_references(contents)

    if not articles_references:
        return html.Div(['Error al procesar el archivo o ninguna referencia encontrada.']), elements, stats_html

    G = build_bibliographic_coupling_network(articles_references, threshold)

    elements, num_communities = nx_to_cytoscape(G, node_size_based, community_algorithm)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes > 0:
        degrees = dict(G.degree())
        max_degree_paper = max(degrees.items(), key=lambda x: x[1]) if degrees else ("N/A", 0)
        edge_weights = [data.get('weight', 0) for _, _, data in G.edges(data=True)]
        max_weight = max(edge_weights) if edge_weights else 0

        stats_html = html.Div([
            html.H4("Estadísticas de la Red"),
            html.P(f"Número de artículos (nodos): {num_nodes}"),
            html.P(f"Número de vínculos de acoplamiento (aristas): {num_edges}"),
            html.P(f"Artículo con más acoplamientos: {max_degree_paper[0]} ({max_degree_paper[1]} conexiones)"),
            html.P(f"Máximo de referencias compartidas entre un par: {max_weight}"),
            html.P(f"Umbral mínimo de referencias compartidas: {threshold}"),
            html.P(f"Número de comunidades detectadas (Leiden): {num_communities}")
        ])

    return html.Div([f'Archivo procesado: {filename}', html.Br(), f'Se encontraron {len(articles_references)} artículos.']), elements, stats_html

# Callback para actualizar el layout
@app.callback(
    Output('cytoscape-network', 'layout'),
    [Input('layout-dropdown', 'value')]
)
def update_layout(layout_value):
    return {'name': layout_value}

@app.callback(
    Output('edge-info', 'children'),
    [Input('cytoscape-network', 'selectedEdgeData'),
     Input('pair-to-articles', 'children'),
     Input('upload-data', 'contents')]
)
def display_edge_info(selectedEdgeData, pair_to_articles_str, contents):
    if not selectedEdgeData or not pair_to_articles_str or not contents:
        return ""
    pair_to_articles = ast.literal_eval(pair_to_articles_str)
    edge = selectedEdgeData[0]
    source = edge['source']
    target = edge['target']
    pair = tuple(sorted([source, target]))
    article_indices = pair_to_articles.get(pair, [])
    if not article_indices:
        return f"No hay artículos compartidos entre {source} y {target}."
    # Optionally, show article indices or extract more info
    return html.Div([
        html.H4(f"Artículos compartidos entre {source} y {target}:"),
        html.Ul([html.Li(f"Artículo #{idx+1}") for idx in article_indices])
    ])

if __name__ == '__main__':
    app.run(debug=True)