#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Enhanced graph construction for GraphTransGeo++

import torch
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.utils import to_undirected, add_self_loops
import math

class EnhancedGraphBuilder:
    """
    Enhanced graph builder with multiple construction strategies
    """
    def __init__(self, use_geographic=True, use_multi_scale=True, use_edge_weights=True):
        self.use_geographic = use_geographic
        self.use_multi_scale = use_multi_scale
        self.use_edge_weights = use_edge_weights
    
    def build_graph(self, x, item, k=5):
        """
        Build enhanced graph with multiple strategies
        
        Args:
            x: Node features [num_nodes, features]
            item: Data item containing additional information
            k: Base number of nearest neighbors
            
        Returns:
            edge_index: Edge index tensor [2, num_edges]
            edge_attr: Edge attributes (optional)
        """
        num_nodes = x.size(0)
        
        # Create base graph
        base_edge_index = self._create_base_graph(x, item, k)
        
        # Add geographic connections if enabled
        if self.use_geographic and 'lm_Y' in item:
            geo_edge_index = self._create_geographic_graph(item['lm_Y'], k)
            edge_index = torch.cat([base_edge_index, geo_edge_index], dim=1)
        else:
            edge_index = base_edge_index
        
        # Add multi-scale connections if enabled
        if self.use_multi_scale and num_nodes > 10:
            multi_scale_edge_index = self._create_multi_scale_graph(x, num_nodes)
            edge_index = torch.cat([edge_index, multi_scale_edge_index], dim=1)
        
        # Make the graph undirected
        edge_index = to_undirected(edge_index)
        
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index)
        
        # Create edge weights if enabled
        if self.use_edge_weights and 'lm_Y' in item:
            edge_attr = self._create_edge_weights(edge_index, item['lm_Y'])
            return edge_index, edge_attr
        
        return edge_index
    
    def _create_base_graph(self, x, item, k=5):
        """
        Create base graph using adaptive k-NN or network topology
        
        Args:
            x: Node features [num_nodes, features]
            item: Data item containing additional information
            k: Base number of nearest neighbors
            
        Returns:
            edge_index: Edge index tensor [2, num_edges]
        """
        num_nodes = x.size(0)
        
        # Adaptive k-NN: adjust k based on node density
        if num_nodes > 20:
            adaptive_k = min(k, num_nodes // 4)
        else:
            adaptive_k = min(k, num_nodes - 1)
        
        # Use network topology if available
        if 'router' in item and isinstance(item['router'], np.ndarray):
            # Create edges based on router connections
            edge_index = self._create_topology_graph(item['router'])
        else:
            # Fallback to improved k-NN
            # Compute cosine similarity instead of Euclidean distance
            x_norm = F.normalize(x, p=2, dim=1)
            similarity = torch.mm(x_norm, x_norm.transpose(0, 1))
            
            # Create k-NN graph based on similarity
            _, indices = torch.topk(similarity, k=adaptive_k+1, dim=1)
            indices = indices[:, 1:]  # Exclude self
            
            # Create edge index
            rows = torch.arange(num_nodes).reshape(-1, 1).repeat(1, adaptive_k).reshape(-1)
            cols = indices.reshape(-1)
            
            edge_index = torch.stack([rows, cols], dim=0)
            
            # Add global connections to improve information flow
            if num_nodes > 10:
                # Connect each node to a few random nodes
                num_global = min(3, num_nodes // 5)
                global_rows = []
                global_cols = []
                
                for i in range(num_nodes):
                    # Select random nodes excluding self
                    candidates = list(range(num_nodes))
                    candidates.remove(i)
                    global_nodes = random.sample(candidates, num_global)
                    
                    for j in global_nodes:
                        global_rows.append(i)
                        global_cols.append(j)
                
                global_edge_index = torch.tensor([global_rows, global_cols], dtype=torch.long)
                
                # Combine with k-NN edges
                edge_index = torch.cat([edge_index, global_edge_index], dim=1)
        
        return edge_index
    
    def _create_topology_graph(self, router_data):
        """
        Create graph based on router topology
        
        Args:
            router_data: Router connection data
            
        Returns:
            edge_index: Edge index tensor [2, num_edges]
        """
        # Extract router connections
        edges = []
        
        # Process router data to extract connections
        if router_data.ndim == 2:
            # If router_data is a 2D array, assume it's an adjacency matrix
            num_nodes = router_data.shape[0]
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if router_data[i, j] > 0:
                        edges.append([i, j])
        else:
            # If router_data is not a 2D array, create a simple fully connected graph
            num_nodes = len(router_data)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edges.append([i, j])
        
        # If no edges were created, add a dummy edge
        if len(edges) == 0:
            edges = [[0, 0]]
        
        # Convert to tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        return edge_index
    
    def _create_geographic_graph(self, coords, k=5):
        """
        Create graph based on geographic proximity
        
        Args:
            coords: Node coordinates [num_nodes, 1, 2]
            k: Number of nearest neighbors
            
        Returns:
            edge_index: Edge index tensor [2, num_edges]
        """
        # Reshape coordinates if needed
        if len(coords.shape) == 3:
            coords = coords.reshape(-1, 2)
        
        # Convert to tensor if needed
        if isinstance(coords, np.ndarray):
            coords = torch.tensor(coords, dtype=torch.float)
        
        num_nodes = coords.shape[0]
        
        # If there are too few nodes, create a fully connected graph
        if num_nodes <= k + 1:
            # Create a fully connected graph (excluding self-loops)
            edge_index = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edge_index.append([i, j])
            
            if len(edge_index) == 0:
                # If there are no edges, create a dummy edge to avoid errors
                edge_index = [[0, 0]]
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            return edge_index
        
        # Compute pairwise distances using Haversine formula
        distances = torch.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # Calculate Haversine distance
                    lon1, lat1 = coords[i]
                    lon2, lat2 = coords[j]
                    distances[i, j] = self._haversine_distance(lon1, lat1, lon2, lat2)
                else:
                    distances[i, j] = float('inf')  # Avoid self-loops
        
        # Create k-NN graph based on geographic proximity
        _, indices = torch.topk(distances, k=k, dim=1, largest=False)
        
        # Create edge index
        rows = torch.arange(num_nodes).reshape(-1, 1).repeat(1, k).reshape(-1)
        cols = indices.reshape(-1)
        
        edge_index = torch.stack([rows, cols], dim=0)
        
        return edge_index
    
    def _create_multi_scale_graph(self, x, num_nodes):
        """
        Create multi-scale graph with connections at different scales
        
        Args:
            x: Node features [num_nodes, features]
            num_nodes: Number of nodes
            
        Returns:
            edge_index: Edge index tensor [2, num_edges]
        """
        # Create edges at different scales
        edges = []
        
        # Scale 1: Connect nodes with similar features (already done in base graph)
        
        # Scale 2: Connect nodes with complementary features
        x_norm = F.normalize(x, p=2, dim=1)
        dissimilarity = 1 - torch.mm(x_norm, x_norm.transpose(0, 1))
        
        # Connect each node to a few nodes with complementary features
        num_complementary = min(3, num_nodes // 5)
        _, indices = torch.topk(dissimilarity, k=num_complementary+1, dim=1)
        indices = indices[:, 1:]  # Exclude self
        
        # Create edge index for complementary connections
        rows = torch.arange(num_nodes).reshape(-1, 1).repeat(1, num_complementary).reshape(-1)
        cols = indices.reshape(-1)
        
        for i in range(rows.size(0)):
            edges.append([rows[i].item(), cols[i].item()])
        
        # Scale 3: Connect nodes with medium similarity
        similarity = torch.mm(x_norm, x_norm.transpose(0, 1))
        medium_similarity = (similarity > 0.3) & (similarity < 0.7)
        
        # Limit the number of medium similarity connections
        max_medium = min(5, num_nodes // 3)
        for i in range(num_nodes):
            medium_nodes = torch.where(medium_similarity[i])[0]
            if len(medium_nodes) > max_medium:
                medium_nodes = medium_nodes[:max_medium]
            
            for j in medium_nodes:
                if i != j:
                    edges.append([i, j.item()])
        
        # Convert to tensor
        if len(edges) == 0:
            # If no edges were created, add a dummy edge
            edges = [[0, 0]]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        return edge_index
    
    def _create_edge_weights(self, edge_index, coords):
        """
        Create edge weights based on geographic distance
        
        Args:
            edge_index: Edge index tensor [2, num_edges]
            coords: Node coordinates [num_nodes, 1, 2]
            
        Returns:
            edge_attr: Edge weights [num_edges, 1]
        """
        # Reshape coordinates if needed
        if len(coords.shape) == 3:
            coords = coords.reshape(-1, 2)
        
        # Convert to tensor if needed
        if isinstance(coords, np.ndarray):
            coords = torch.tensor(coords, dtype=torch.float)
        
        # Calculate weights for each edge
        num_edges = edge_index.size(1)
        edge_attr = torch.zeros(num_edges, 1)
        
        for i in range(num_edges):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src != dst:  # Skip self-loops
                # Calculate Haversine distance
                lon1, lat1 = coords[src]
                lon2, lat2 = coords[dst]
                dist = self._haversine_distance(lon1, lat1, lon2, lat2)
                
                # Convert distance to edge weight (inverse of distance)
                # Use a smooth decay function
                weight = 1.0 / (1.0 + dist)
            else:
                # Self-loop weight
                weight = 1.0
            
            edge_attr[i, 0] = weight
        
        return edge_attr
    
    def _haversine_distance(self, lon1, lat1, lon2, lat2):
        """
        Calculate Haversine distance between two points
        
        Args:
            lon1, lat1: Coordinates of first point
            lon2, lat2: Coordinates of second point
            
        Returns:
            distance: Distance in kilometers
        """
        # Convert to radians
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        return c * r


class DynamicGraphBuilder(torch.nn.Module):
    """
    Dynamic graph builder that can update the graph during training
    """
    def __init__(self, initial_builder, update_interval=5):
        super(DynamicGraphBuilder, self).__init__()
        self.initial_builder = initial_builder
        self.update_interval = update_interval
        self.update_count = 0
        
        # Learnable parameters for graph update
        self.feature_weight = torch.nn.Parameter(torch.tensor(0.7))
        self.geographic_weight = torch.nn.Parameter(torch.tensor(0.3))
        self.global_weight = torch.nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x, item, edge_index=None, k=5):
        """
        Build or update graph dynamically
        
        Args:
            x: Node features [num_nodes, features]
            item: Data item containing additional information
            edge_index: Current edge index (if None, create new graph)
            k: Base number of nearest neighbors
            
        Returns:
            edge_index: Updated edge index tensor [2, num_edges]
            edge_attr: Updated edge attributes (optional)
        """
        # If no current edge_index or update interval reached, create new graph
        if edge_index is None or self.update_count % self.update_interval == 0:
            # Create new graph
            result = self.initial_builder.build_graph(x, item, k)
            self.update_count += 1
            return result
        
        # Otherwise, update existing graph
        num_nodes = x.size(0)
        
        # Compute feature similarity
        x_norm = F.normalize(x, p=2, dim=1)
        similarity = torch.mm(x_norm, x_norm.transpose(0, 1))
        
        # Get existing edges
        existing_edges = set()
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            existing_edges.add((src, dst))
        
        # Add new edges based on current features
        new_edges = []
        adaptive_k = min(k, num_nodes // 4) if num_nodes > 20 else min(k, num_nodes - 1)
        
        _, indices = torch.topk(similarity, k=adaptive_k+1, dim=1)
        indices = indices[:, 1:]  # Exclude self
        
        for i in range(num_nodes):
            for j in indices[i]:
                j = j.item()
                if i != j and (i, j) not in existing_edges:
                    # Add with probability based on feature weight
                    if random.random() < self.feature_weight:
                        new_edges.append([i, j])
                        existing_edges.add((i, j))
        
        # Add geographic edges if available
        if 'lm_Y' in item and self.geographic_weight > 0:
            coords = item['lm_Y']
            if len(coords.shape) == 3:
                coords = coords.reshape(-1, 2)
            
            if isinstance(coords, np.ndarray):
                coords = torch.tensor(coords, dtype=torch.float)
            
            # Compute pairwise distances
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j and (i, j) not in existing_edges:
                        # Calculate Haversine distance
                        lon1, lat1 = coords[i]
                        lon2, lat2 = coords[j]
                        dist = self.initial_builder._haversine_distance(lon1, lat1, lon2, lat2)
                        
                        # Add edge with probability inversely proportional to distance
                        prob = self.geographic_weight / (1.0 + dist)
                        if random.random() < prob:
                            new_edges.append([i, j])
                            existing_edges.add((i, j))
        
        # Add some global connections
        if self.global_weight > 0:
            num_global = min(3, num_nodes // 5)
            for i in range(num_nodes):
                # Select random nodes excluding self
                candidates = list(range(num_nodes))
                candidates.remove(i)
                global_nodes = random.sample(candidates, num_global)
                
                for j in global_nodes:
                    if (i, j) not in existing_edges:
                        # Add with probability based on global weight
                        if random.random() < self.global_weight:
                            new_edges.append([i, j])
                            existing_edges.add((i, j))
        
        # Combine with existing edges
        if new_edges:
            new_edge_index = torch.tensor(new_edges, dtype=torch.long).t()
            updated_edge_index = torch.cat([edge_index, new_edge_index], dim=1)
        else:
            updated_edge_index = edge_index
        
        # Make the graph undirected
        updated_edge_index = to_undirected(updated_edge_index)
        
        # Add self-loops
        updated_edge_index, _ = add_self_loops(updated_edge_index)
        
        # Create edge weights if enabled
        if self.initial_builder.use_edge_weights and 'lm_Y' in item:
            edge_attr = self.initial_builder._create_edge_weights(updated_edge_index, item['lm_Y'])
            return updated_edge_index, edge_attr
        
        self.update_count += 1
        return updated_edge_index
