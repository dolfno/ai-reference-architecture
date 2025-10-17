import React from 'react';
import { X, Tag, Network } from 'lucide-react';
import { loadArchitectureComponents, loadCategories } from '../utils/storageUtils';
import './ComponentDetail.css';

const ComponentDetail = ({ componentId, onClose, onNavigate }) => {
  if (!componentId) return null;

  // Load data dynamically each time component renders
  const architectureComponents = loadArchitectureComponents();
  const categories = loadCategories();

  const component = architectureComponents[componentId];
  if (!component) return null;

  const category = categories[component.category];

  // Format the details text with markdown-like styling
  const formatDetails = (text) => {
    return text.split('\n').map((line, index) => {
      // Headers
      if (line.startsWith('**') && line.endsWith('**')) {
        return (
          <h3 key={index} className="detail-subheading">
            {line.replace(/\*\*/g, '')}
          </h3>
        );
      }

      // Bold inline
      if (line.includes('**')) {
        const parts = line.split(/(\*\*.*?\*\*)/g);
        return (
          <p key={index}>
            {parts.map((part, i) => {
              if (part.startsWith('**') && part.endsWith('**')) {
                return <strong key={i}>{part.replace(/\*\*/g, '')}</strong>;
              }
              return part;
            })}
          </p>
        );
      }

      // Bullet points
      if (line.trim().startsWith('-')) {
        return (
          <li key={index} className="detail-list-item">
            {line.trim().substring(1).trim()}
          </li>
        );
      }

      // Code blocks (lines starting with numbers followed by .)
      if (/^\d+\./.test(line.trim())) {
        return (
          <li key={index} className="detail-numbered-item">
            {line.trim().replace(/^\d+\.\s*/, '')}
          </li>
        );
      }

      // Empty lines
      if (line.trim() === '') {
        return <div key={index} className="detail-spacer" />;
      }

      // Regular paragraphs
      return <p key={index}>{line}</p>;
    });
  };

  return (
    <div className="component-detail-overlay" onClick={onClose}>
      <div className="component-detail-panel" onClick={(e) => e.stopPropagation()}>
        <button className="close-button" onClick={onClose}>
          <X size={24} />
        </button>

        <div className="detail-header" style={{ borderLeftColor: category.color }}>
          <div className="detail-category" style={{ color: category.color }}>
            <Tag size={16} />
            {category.name}
          </div>
          <h1 className="detail-title">{component.title}</h1>
          <p className="detail-subtitle">{component.description}</p>
        </div>

        <div className="detail-content">
          <div className="detail-body">
            {formatDetails(component.details)}
          </div>

          {component.connections && component.connections.length > 0 && (
            <div className="detail-connections">
              <h3 className="connections-title">
                <Network size={20} />
                Related Components
              </h3>
              <div className="connections-grid">
                {component.connections.map(connId => {
                  const connectedComp = architectureComponents[connId];
                  const connCategory = categories[connectedComp.category];
                  return (
                    <button
                      key={connId}
                      className="connection-card"
                      style={{
                        borderLeftColor: connCategory.color,
                        background: `linear-gradient(135deg, ${connCategory.color}15, ${connCategory.color}05)`
                      }}
                      onClick={() => onNavigate(connId)}
                    >
                      <div className="connection-name">{connectedComp.title}</div>
                      <div className="connection-category">{connCategory.name}</div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ComponentDetail;
