"""
File I/O utilities for QAFD-RAG.

Provides JSON and XML file operations.
"""

import json
import os
import xml.etree.ElementTree as ET


def load_json(file_name: str):
    """
    Load JSON data from a file.

    Parameters:
    -----------
    file_name : str
        Path to the JSON file

    Returns:
    --------
    dict or None
        Parsed JSON data, or None if file doesn't exist
    """
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name: str):
    """
    Write JSON data to a file.

    Parameters:
    -----------
    json_obj : Any
        Data to write as JSON
    file_name : str
        Path to the output file
    """
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def save_data_to_file(data, file_name: str):
    """
    Save data to a JSON file with pretty formatting.

    Parameters:
    -----------
    data : Any
        Data to save
    file_name : str
        Path to the output file
    """
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def xml_to_json(xml_file: str):
    """
    Convert a GraphML XML file to JSON format.

    Parameters:
    -----------
    xml_file : str
        Path to the GraphML file

    Returns:
    --------
    dict or None
        Dictionary with 'nodes' and 'edges' lists, or None on error
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        print(f"Root element: {root.tag}")
        print(f"Root attributes: {root.attrib}")

        data = {"nodes": [], "edges": []}
        namespace = {"": "http://graphml.graphdrawing.org/xmlns"}

        for node in root.findall(".//node", namespace):
            node_data = {
                "id": node.get("id").strip('"'),
                "entity_type": node.find("./data[@key='d0']", namespace).text.strip('"')
                if node.find("./data[@key='d0']", namespace) is not None
                else "",
                "description": node.find("./data[@key='d1']", namespace).text
                if node.find("./data[@key='d1']", namespace) is not None
                else "",
                "source_id": node.find("./data[@key='d2']", namespace).text
                if node.find("./data[@key='d2']", namespace) is not None
                else "",
            }
            data["nodes"].append(node_data)

        for edge in root.findall(".//edge", namespace):
            edge_data = {
                "source": edge.get("source").strip('"'),
                "target": edge.get("target").strip('"'),
                "weight": float(edge.find("./data[@key='d3']", namespace).text)
                if edge.find("./data[@key='d3']", namespace) is not None
                else 0.0,
                "description": edge.find("./data[@key='d4']", namespace).text
                if edge.find("./data[@key='d4']", namespace) is not None
                else "",
                "keywords": edge.find("./data[@key='d5']", namespace).text
                if edge.find("./data[@key='d5']", namespace) is not None
                else "",
                "source_id": edge.find("./data[@key='d6']", namespace).text
                if edge.find("./data[@key='d6']", namespace) is not None
                else "",
            }
            data["edges"].append(edge_data)

        print(f"Found {len(data['nodes'])} nodes and {len(data['edges'])} edges")

        return data
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


__all__ = [
    "load_json",
    "write_json",
    "save_data_to_file",
    "xml_to_json",
]
