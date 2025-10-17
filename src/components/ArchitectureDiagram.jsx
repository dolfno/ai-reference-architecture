import React, { useCallback, useMemo } from 'react';
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { architectureComponents, categories } from '../data/architectureData';
import './ArchitectureDiagram.css';

const ArchitectureDiagram = ({ onNodeClick }) => {
  // Create nodes from architecture data
  const initialNodes = useMemo(() => {
    const nodesByCategory = {
      core: [],
      advanced: [],
      infrastructure: []
    };

    // Group components by category
    Object.values(architectureComponents).forEach(comp => {
      nodesByCategory[comp.category].push(comp);
    });

    const nodes = [];
    let yOffset = 0;

    // Create nodes for each category
    Object.entries(nodesByCategory).forEach(([categoryId, components], categoryIndex) => {
      const category = categories[categoryId];
      const xSpacing = 280;
      const ySpacing = 120;
      const componentsPerRow = 4;

      components.forEach((comp, index) => {
        const row = Math.floor(index / componentsPerRow);
        const col = index % componentsPerRow;

        nodes.push({
          id: comp.id,
          type: 'default',
          data: {
            label: (
              <div className="node-content">
                <div className="node-title">{comp.title}</div>
                <div className="node-description">{comp.description}</div>
              </div>
            )
          },
          position: {
            x: col * xSpacing,
            y: yOffset + row * ySpacing + 100
          },
          style: {
            background: category.color,
            color: 'white',
            border: '2px solid rgba(255, 255, 255, 0.3)',
            borderRadius: '10px',
            padding: '15px',
            width: 240,
            fontSize: '12px',
            cursor: 'pointer',
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
          },
        });
      });

      // Add category label
      nodes.push({
        id: `category-${categoryId}`,
        type: 'default',
        data: {
          label: (
            <div className="category-label">
              <strong>{category.name}</strong>
            </div>
          )
        },
        position: { x: 0, y: yOffset },
        style: {
          background: category.color,
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          padding: '10px 20px',
          fontSize: '16px',
          fontWeight: 'bold',
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)',
          pointerEvents: 'none',
        },
        draggable: false,
      });

      // Calculate offset for next category
      const rows = Math.ceil(components.length / componentsPerRow);
      yOffset += (rows * ySpacing) + 200;
    });

    return nodes;
  }, []);

  // Create edges based on connections
  const initialEdges = useMemo(() => {
    const edges = [];

    Object.values(architectureComponents).forEach(comp => {
      if (comp.connections) {
        comp.connections.forEach(targetId => {
          edges.push({
            id: `${comp.id}-${targetId}`,
            source: comp.id,
            target: targetId,
            type: 'smoothstep',
            animated: true,
            style: {
              stroke: 'rgba(255, 255, 255, 0.4)',
              strokeWidth: 2,
            },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: 'rgba(255, 255, 255, 0.4)',
            },
          });
        });
      }
    });

    return edges;
  }, []);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onNodeClickHandler = useCallback((event, node) => {
    // Don't trigger for category labels
    if (node.id.startsWith('category-')) return;

    if (onNodeClick) {
      onNodeClick(node.id);
    }
  }, [onNodeClick]);

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClickHandler}
        fitView
        minZoom={0.1}
        maxZoom={1.5}
        defaultViewport={{ x: 0, y: 0, zoom: 0.8 }}
      >
        <Controls />
        <MiniMap
          nodeColor={(node) => {
            if (node.id.startsWith('category-')) return node.style.background;
            const comp = architectureComponents[node.id];
            return comp ? categories[comp.category].color : '#ccc';
          }}
          maskColor="rgba(0, 0, 0, 0.2)"
        />
        <Background variant="dots" gap={12} size={1} color="rgba(255, 255, 255, 0.2)" />
      </ReactFlow>
    </div>
  );
};

export default ArchitectureDiagram;
