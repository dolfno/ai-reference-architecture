import React, { useState } from 'react';
import ArchitectureDiagram from './components/ArchitectureDiagram';
import ComponentDetail from './components/ComponentDetail';
import { Brain, Github, BookOpen } from 'lucide-react';
import './App.css';

function App() {
  const [selectedComponent, setSelectedComponent] = useState(null);

  const handleNodeClick = (componentId) => {
    setSelectedComponent(componentId);
  };

  const handleCloseDetail = () => {
    setSelectedComponent(null);
  };

  const handleNavigate = (componentId) => {
    setSelectedComponent(componentId);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="header-title">
            <Brain size={32} />
            <h1>GenAI Architecture Reference</h1>
          </div>
          <p className="header-subtitle">
            Interactive guide to understanding the components, patterns, and infrastructure of Generative AI applications
          </p>
        </div>
        <div className="header-actions">
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="header-link"
          >
            <Github size={20} />
            <span>GitHub</span>
          </a>
        </div>
      </header>

      <div className="app-content">
        <aside className="sidebar">
          <div className="sidebar-section">
            <h3 className="sidebar-title">
              <BookOpen size={18} />
              How to Use
            </h3>
            <ul className="sidebar-list">
              <li>Click on any component to view detailed information</li>
              <li>Explore connections between related components</li>
              <li>Use the minimap to navigate the architecture</li>
              <li>Zoom and pan to explore different areas</li>
            </ul>
          </div>

          <div className="sidebar-section">
            <h3 className="sidebar-title">Categories</h3>
            <div className="category-legend">
              <div className="legend-item">
                <span className="legend-color" style={{ background: '#3B82F6' }}></span>
                <span>Core Components</span>
              </div>
              <div className="legend-item">
                <span className="legend-color" style={{ background: '#8B5CF6' }}></span>
                <span>Advanced Patterns</span>
              </div>
              <div className="legend-item">
                <span className="legend-color" style={{ background: '#10B981' }}></span>
                <span>Infrastructure</span>
              </div>
            </div>
          </div>

          <div className="sidebar-section">
            <h3 className="sidebar-title">Coverage</h3>
            <ul className="sidebar-list small">
              <li>LLMs & Embeddings</li>
              <li>Vector Databases</li>
              <li>RAG & Semantic Search</li>
              <li>AI Agents & Orchestration</li>
              <li>Fine-tuning & Evaluation</li>
              <li>Deployment & Scaling</li>
              <li>Security & Monitoring</li>
            </ul>
          </div>
        </aside>

        <main className="main-content">
          <div className="diagram-container">
            <ArchitectureDiagram onNodeClick={handleNodeClick} />
          </div>
        </main>
      </div>

      <ComponentDetail
        componentId={selectedComponent}
        onClose={handleCloseDetail}
        onNavigate={handleNavigate}
      />

      <footer className="app-footer">
        <p>
          Built to help developers understand and implement GenAI architectures.
          Click any component above to learn more.
        </p>
      </footer>
    </div>
  );
}

export default App;
