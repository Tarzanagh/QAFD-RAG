#!/usr/bin/env python3
"""
Enhanced Interactive Knowledge Graph Visualizer Module for QAFD_RAG
Creates an HTML file with interactive graph showing all nodes and relationships
with improved color schemes for different components

This module is designed to be imported by the main QAFD_RAG script.
"""

import networkx as nx
from pyvis.network import Network
import json
from pathlib import Path

def create_interactive_kg_visualization(graphml_path, output_html="knowledge_graph_interactive.html"):
    """
    Create an interactive HTML visualization of the knowledge graph with enhanced colors
    
    Args:
        graphml_path (str): Path to the GraphML file
        output_html (str): Output HTML file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Loading graph from: {graphml_path}")
    
    # Load the GraphML file
    try:
        G = nx.read_graphml(graphml_path)
        print(f"✅ Graph loaded successfully!")
        print(f"   Nodes: {G.number_of_nodes()}")
        print(f"   Edges: {G.number_of_edges()}")
        
        # ADDED: Debug edges
        if G.number_of_edges() > 0:
            print("Sample edges:")
            for i, (source, target, data) in enumerate(G.edges(data=True)):
                if i < 3:  # Show first 3 edges
                    print(f"  {source} -> {target}: {data}")
        else:
            print("WARNING: No edges found in GraphML file!")
            
    except Exception as e:
        print(f"❌ Error loading graph: {e}")
        return False
    
    # Create Pyvis network with enhanced styling
    net = Network(
        height="900px", 
        width="100%", 
        bgcolor="#0a0a0a",  # Darker background for better contrast
        font_color="white",
        directed=False,
        notebook=False
    )
    
    # Enhanced color schemes
    node_colors = {
        'entity': {
            'color': '#FF4757',      # Bright red for entities
            'border': '#FF3742',
            'highlight': '#FF6B7A'
        },
        'chunk': {
            'color': '#2ED573',      # Bright green for chunks
            'border': '#20BF6B',
            'highlight': '#54E091'
        },
        'table': {
            'color': '#3742FA',      # Electric blue for tables
            'border': '#2F3542',
            'highlight': '#5352ED'
        },
        'column': {
            'color': '#FFA726',      # Orange for columns
            'border': '#FF9800',
            'highlight': '#FFB74D'
        },
        'complete_table': {          # ADDED: Handle complete_table type
            'color': '#3742FA',      # Same as table
            'border': '#2F3542',
            'highlight': '#5352ED'
        },
        'default': {
            'color': '#A4B0BE',      # Light gray for others
            'border': '#747D8C',
            'highlight': '#DDD6FE'
        }
    }
    
    # Edge color schemes
    edge_colors = {
        'primary_key': '#FF6B35',     # Orange-red for primary key edges
        'foreign_key': '#4834D4',     # Purple for foreign key edges
        'contains': '#00D2D3',        # Cyan for containment relationships
        'belongs_to': '#FF9FF3',      # Pink for belongs_to relationships
        'references': '#54A0FF',      # Light blue for references
        'has': '#5F27CD',             # Deep purple for has relationships
        'table_structure': '#00D2D3', # Same as contains
        'default': '#57606F'          # Gray for standard edges
    }
    
    print("Adding nodes with enhanced styling...")
    # First, let's analyze what node types we actually have
    actual_node_types = set()
    entity_types = set()
    for node_id, node_data in G.nodes(data=True):
        node_type = node_data.get('type', 'unknown')
        entity_type = node_data.get('entity_type', 'unknown')
        actual_node_types.add(node_type)
        entity_types.add(entity_type)
    
    print(f"Detected 'type' attributes: {actual_node_types}")
    print(f"Detected 'entity_type' attributes: {entity_types}")
    
    for node_id, node_data in G.nodes(data=True):
        # Read the entity_type directly from the data
        entity_type = node_data.get('entity_type', '').lower()
        node_type = node_data.get('type', '').lower()
        
        # Use entity_type as the primary classifier
        if entity_type in ['table', 'complete_table']:
            detected_type = 'table'
        elif entity_type == 'column':
            detected_type = 'column'
        elif entity_type in ['entity', 'chunk']:
            detected_type = entity_type
        elif node_type in ['table', 'column', 'entity', 'chunk']:
            detected_type = node_type
        else:
            detected_type = 'default'
        
        color_scheme = node_colors.get(detected_type, node_colors['default'])
        
        # Create detailed hover info
        title = f"<b>Node ID:</b> {node_id}<br>"
        # title += f"<b>Type:</b> {detected_type.title()}<br>"
        
        # Show entity_type if it exists
        if 'entity_type' in node_data:
            title += f"<b>Entity Type:</b> {node_data['entity_type']}<br>"
        
        # Add all other attributes
        for key, value in node_data.items():
            if key not in ['type', 'entity_type']:
                # Truncate long values for readability
                display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                title += f"<b>{key.title()}:</b> {display_value}<br>"
        
        # Determine node size based on detected type
        size_mapping = {
            'table': 35,      # Largest for tables
            'column': 25,     # Medium for columns
            'entity': 30,     # Large for entities
            'chunk': 20,      # Smaller for chunks
            'default': 18     # Smallest for others
        }
        
        # Add node with enhanced styling
        net.add_node(
            str(node_id),
            label=str(node_id)[:25] + ("..." if len(str(node_id)) > 25 else ""),
            title=title,
            color={
                'background': color_scheme['color'],
                'border': color_scheme['border'],
                'highlight': {
                    'background': color_scheme['highlight'],
                    'border': color_scheme['border']
                }
            },
            size=size_mapping.get(detected_type, size_mapping['default']),
            font={'size': 14, 'color': 'white', 'face': 'arial'},
            borderWidth=3,
            shadow={'enabled': True, 'color': 'rgba(0,0,0,0.5)', 'size': 10}
        )
    
    print("Adding edges with relationship-based coloring...")
    # First, analyze actual edge types
    actual_edge_types = set()
    for source, target, edge_data in G.edges(data=True):
        # Try multiple ways to get relationship info
        relationship = (
            edge_data.get('relationship') or
            edge_data.get('description') or
            edge_data.get('keywords') or
            edge_data.get('label') or
            'default'
        )
        actual_edge_types.add(str(relationship).lower())
    
    print(f"Detected edge types: {actual_edge_types}")
    
    edges_added = 0
    for source, target, edge_data in G.edges(data=True):
        # Try to get meaningful relationship info
        relationship = (
            edge_data.get('relationship') or
            edge_data.get('description') or
            edge_data.get('keywords') or
            edge_data.get('label') or
            'default'
        )
        
        # Convert to string and normalize
        relationship_str = str(relationship).lower()
        
        # Determine edge color and width based on relationship type
        edge_color = edge_colors['default']
        edge_width = 2
        
        # Enhanced relationship detection with keyword matching
        if 'foreign_key' in relationship_str or 'references' in relationship_str:
            edge_color = edge_colors['foreign_key']
            edge_width = 3
        elif 'primary_key' in relationship_str:
            edge_color = edge_colors['primary_key']
            edge_width = 4
        elif 'contains' in relationship_str or 'table_structure' in relationship_str:
            edge_color = edge_colors['contains']
            edge_width = 3
        elif 'belongs_to' in relationship_str:
            edge_color = edge_colors['belongs_to']
            edge_width = 2
        elif 'has' in relationship_str:
            edge_color = edge_colors['has']
            edge_width = 2
        else:
            # For any other specific relationships
            edge_color = '#FFD93D'  # Bright yellow for other relationships
            edge_width = 2
        
        # Create detailed edge title
        title = f"<b>Relationship:</b> {relationship}<br>"
        title += f"<b>From:</b> {source}<br>"
        title += f"<b>To:</b> {target}<br>"
        
        # Add other edge attributes
        for key, value in edge_data.items():
            if key not in ['relationship', 'label', 'description', 'keywords']:
                title += f"<b>{key.title()}:</b> {str(value)}<br>"
        
        # Add edge with enhanced styling
        net.add_edge(
            str(source),
            str(target),
            label=str(relationship)[:25] if relationship != 'default' else '',  # Show label only if meaningful
            title=title,
            color={
                'color': edge_color,
                'highlight': edge_color,
                'hover': edge_color,
                'opacity': 0.8
            },
            width=edge_width,
            font={'size': 11, 'color': 'white'}
        )
        edges_added += 1

    print(f"Successfully added {edges_added} edges to visualization")
    
    # Enhanced physics and layout configuration
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {
          "iterations": 300,
          "updateInterval": 25
        },
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.2,
          "springLength": 120,
          "springConstant": 0.05,
          "damping": 0.15,
          "avoidOverlap": 0.2
        }
      },
      "nodes": {
        "font": {
          "size": 14,
          "color": "white",
          "face": "arial",
          "strokeWidth": 2,
          "strokeColor": "black"
        },
        "borderWidth": 3,
        "shadow": {
          "enabled": true,
          "color": "rgba(0,0,0,0.5)",
          "size": 10
        },
        "chosen": {
          "node": true,
          "label": true
        }
      },
      "edges": {
        "font": {
          "size": 11,
          "color": "white",
          "face": "arial",
          "strokeWidth": 1,
          "strokeColor": "black"
        },
        "smooth": {
          "type": "continuous",
          "roundness": 0.5
        },
        "shadow": {
          "enabled": true,
          "color": "rgba(0,0,0,0.3)"
        },
        "chosen": {
          "edge": true,
          "label": true
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "hideEdgesOnDrag": false,
        "hideNodesOnDrag": false
      },
      "layout": {
        "improvedLayout": true,
        "hierarchical": false
      }
    }
    """)
    
    # Generate and save the HTML
    try:
        print("Generating enhanced HTML visualization...")
        
        html_string = net.generate_html()
        
        # Add custom CSS for better styling
        custom_css = """
        <style>
        body {
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        #mynetworkid {
            border: 2px solid #333;
            border-radius: 10px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        .legend {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(26, 26, 26, 0.9);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #444;
            color: white;
            font-size: 12px;
            max-width: 200px;
            z-index: 1000;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }
        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            margin-right: 10px;
            border: 2px solid rgba(255,255,255,0.3);
        }
        </style>
        """
        
        # Add legend HTML with correct colors
        legend_html = """
        <div class="legend">
            <h4 style="margin-top: 0; color: #fff;">Node Types</h4>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #3742FA;"></div>
                <span>Tables</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #FFA726;"></div>
                <span>Columns</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #FF4757;"></div>
                <span>Entities</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #2ED573;"></div>
                <span>Chunks</span>
            </div>
            <h4 style="color: #fff; margin-bottom: 5px;">Edge Types</h4>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #57606F; border-radius: 2px;"></div>
                <span>Default</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #FF6B35; border-radius: 2px;"></div>
                <span>Primary Key</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #4834D4; border-radius: 2px;"></div>
                <span>Foreign Key</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #00D2D3; border-radius: 2px;"></div>
                <span>Contains</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #FFD93D; border-radius: 2px;"></div>
                <span>Other</span>
            </div>
        </div>
        """
        
        # Insert custom styling into HTML
        html_string = html_string.replace('</head>', f'{custom_css}</head>')
        html_string = html_string.replace('<body>', f'<body>{legend_html}')
        
        # Write enhanced HTML to file
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(html_string)
            
        print(f"✅ Enhanced interactive visualization saved as: {output_html}")
        print(f"🌐 Open this file in your browser to explore the knowledge graph!")
        print(f"🎨 Features enhanced color coding for nodes and edges!")
        
        # Print detailed statistics
        print(f"\n📊 Graph Statistics:")
        print(f"   Total nodes: {G.number_of_nodes()}")
        print(f"   Total edges: {G.number_of_edges()}")
        print(f"   Edges added to visualization: {edges_added}")
        print(f"   Density: {nx.density(G):.4f}")
        
        if G.number_of_nodes() > 0:
            degrees = dict(G.degree())
            max_degree_node = max(degrees, key=degrees.get)
            avg_degree = sum(degrees.values()) / len(degrees)
            print(f"   Most connected node: {max_degree_node} (degree: {degrees[max_degree_node]})")
            print(f"   Average degree: {avg_degree:.2f}")
            
        # Enhanced node types breakdown with actual data attributes
        node_types = {}
        entity_types = {}
        for node_id, data in G.nodes(data=True):
            # Original type from data
            original_type = data.get('type', 'unknown')
            node_types[original_type] = node_types.get(original_type, 0) + 1
            
            # Entity type from data
            entity_type = data.get('entity_type', 'unknown')
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        print(f"\n🏷️  Node 'type' Attribute Distribution:")
        for node_type, count in sorted(node_types.items()):
            percentage = (count / G.number_of_nodes()) * 100
            print(f"   {node_type}: {count} ({percentage:.1f}%)")
        
        print(f"\n🔍 Node 'entity_type' Attribute Distribution:")
        for entity_type, count in sorted(entity_types.items()):
            percentage = (count / G.number_of_nodes()) * 100
            print(f"   {entity_type}: {count} ({percentage:.1f}%)")
        
        # Edge types breakdown
        edge_types = {}
        for _, _, data in G.edges(data=True):
            edge_type = (
                data.get('relationship') or
                data.get('description') or
                data.get('keywords') or
                data.get('label') or
                'unknown'
            )
            edge_types[str(edge_type)] = edge_types.get(str(edge_type), 0) + 1
        
        print(f"\n🔗 Edge Types Distribution:")
        for edge_type, count in sorted(edge_types.items()):
            percentage = (count / G.number_of_edges()) * 100 if G.number_of_edges() > 0 else 0
            print(f"   {edge_type}: {count} ({percentage:.1f}%)")
            
        return True
            
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        
        # Fallback approach
        try:
            print("Trying fallback method...")
            net.show(output_html)
            print(f"✅ Fallback successful! File saved as: {output_html}")
            return True
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return False


# For standalone usage
def main():
    """
    Main function for standalone usage - customize the path to your GraphML file here
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Create interactive knowledge graph visualization")
    parser.add_argument("graphml_path", help="Path to the GraphML file")
    parser.add_argument("--output", "-o", default="knowledge_graph_interactive.html", 
                       help="Output HTML file name (default: knowledge_graph_interactive.html)")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.graphml_path).exists():
        print(f"❌ GraphML file not found: {args.graphml_path}")
        return
    
    # Create visualization
    success = create_interactive_kg_visualization(args.graphml_path, args.output)
    if success:
        print(f"\n🎉 Visualization complete! Open {args.output} in your browser.")
    else:
        print(f"\n💥 Visualization failed. Check the error messages above.")


if __name__ == "__main__":
    main()