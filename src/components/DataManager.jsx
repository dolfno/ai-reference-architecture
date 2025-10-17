import React, { useState, useEffect } from 'react';
import { 
  loadArchitectureComponents, 
  loadCategories,
  updateComponent,
  removeComponent,
  updateCategory,
  removeCategory,
  exportData,
  importData,
  resetToDefaults
} from '../utils/storageUtils.js';
import './DataManager.css';

const DataManager = ({ onDataChange }) => {
  const [components, setComponents] = useState({});
  const [categories, setCategories] = useState({});
  const [activeTab, setActiveTab] = useState('components');
  const [editingComponent, setEditingComponent] = useState(null);
  const [editingCategory, setEditingCategory] = useState(null);
  const [showNewComponent, setShowNewComponent] = useState(false);
  const [showNewCategory, setShowNewCategory] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = () => {
    setComponents(loadArchitectureComponents());
    setCategories(loadCategories());
  };

  const handleUpdateComponent = (component) => {
    if (updateComponent(component)) {
      loadData();
      setEditingComponent(null);
      setShowNewComponent(false);
      if (onDataChange) onDataChange();
    }
  };

  const handleRemoveComponent = (componentId) => {
    if (window.confirm('Are you sure you want to delete this component?')) {
      if (removeComponent(componentId)) {
        loadData();
        if (onDataChange) onDataChange();
      }
    }
  };

  const handleUpdateCategory = (category) => {
    if (updateCategory(category)) {
      loadData();
      setEditingCategory(null);
      setShowNewCategory(false);
      if (onDataChange) onDataChange();
    }
  };

  const handleRemoveCategory = (categoryId) => {
    if (window.confirm('Are you sure you want to delete this category?')) {
      if (removeCategory(categoryId)) {
        loadData();
        if (onDataChange) onDataChange();
      }
    }
  };

  const handleExport = () => {
    const data = exportData();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `architecture-data-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target.result);
          if (importData(data)) {
            loadData();
            if (onDataChange) onDataChange();
            alert('Data imported successfully!');
          } else {
            alert('Failed to import data. Please check the file format.');
          }
        } catch (error) {
          alert('Invalid JSON file.');
        }
      };
      reader.readAsText(file);
    }
  };

  const handleReset = () => {
    if (window.confirm('This will reset all data to defaults. Are you sure?')) {
      resetToDefaults();
      loadData();
      if (onDataChange) onDataChange();
    }
  };

  return (
    <div className="data-manager">
      <div className="data-manager-header">
        <h2>Architecture Data Manager</h2>
        <div className="data-manager-actions">
          <button onClick={handleExport} className="btn btn-secondary">Export Data</button>
          <label className="btn btn-secondary">
            Import Data
            <input type="file" accept=".json" onChange={handleImport} style={{ display: 'none' }} />
          </label>
          <button onClick={handleReset} className="btn btn-danger">Reset to Defaults</button>
        </div>
      </div>

      <div className="tabs">
        <button 
          className={activeTab === 'components' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('components')}
        >
          Components ({Object.keys(components).length})
        </button>
        <button 
          className={activeTab === 'categories' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('categories')}
        >
          Categories ({Object.keys(categories).length})
        </button>
      </div>

      {activeTab === 'components' && (
        <div className="tab-content">
          <div className="section-header">
            <h3>Architecture Components</h3>
            <button 
              onClick={() => setShowNewComponent(true)} 
              className="btn btn-primary"
            >
              Add New Component
            </button>
          </div>

          {showNewComponent && (
            <ComponentForm
              component={{
                id: '',
                title: '',
                category: 'core',
                description: '',
                details: '',
                connections: []
              }}
              categories={categories}
              onSave={handleUpdateComponent}
              onCancel={() => setShowNewComponent(false)}
            />
          )}

          <div className="components-list">
            {Object.values(components).map((component) => (
              <div key={component.id} className="component-item">
                {editingComponent === component.id ? (
                  <ComponentForm
                    component={component}
                    categories={categories}
                    onSave={handleUpdateComponent}
                    onCancel={() => setEditingComponent(null)}
                  />
                ) : (
                  <div className="component-summary">
                    <div className="component-info">
                      <h4>{component.title}</h4>
                      <p><strong>ID:</strong> {component.id}</p>
                      <p><strong>Category:</strong> {component.category}</p>
                      <p><strong>Description:</strong> {component.description}</p>
                      <p><strong>Connections:</strong> {component.connections?.join(', ') || 'None'}</p>
                    </div>
                    <div className="component-actions">
                      <button 
                        onClick={() => setEditingComponent(component.id)}
                        className="btn btn-small btn-secondary"
                      >
                        Edit
                      </button>
                      <button 
                        onClick={() => handleRemoveComponent(component.id)}
                        className="btn btn-small btn-danger"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === 'categories' && (
        <div className="tab-content">
          <div className="section-header">
            <h3>Categories</h3>
            <button 
              onClick={() => setShowNewCategory(true)} 
              className="btn btn-primary"
            >
              Add New Category
            </button>
          </div>

          {showNewCategory && (
            <CategoryForm
              category={{
                id: '',
                name: '',
                color: '#3B82F6',
                description: ''
              }}
              onSave={handleUpdateCategory}
              onCancel={() => setShowNewCategory(false)}
            />
          )}

          <div className="categories-list">
            {Object.values(categories).map((category) => (
              <div key={category.id} className="category-item">
                {editingCategory === category.id ? (
                  <CategoryForm
                    category={category}
                    onSave={handleUpdateCategory}
                    onCancel={() => setEditingCategory(null)}
                  />
                ) : (
                  <div className="category-summary">
                    <div className="category-info">
                      <h4 style={{ color: category.color }}>{category.name}</h4>
                      <p><strong>ID:</strong> {category.id}</p>
                      <p><strong>Color:</strong> {category.color}</p>
                      <p><strong>Description:</strong> {category.description}</p>
                    </div>
                    <div className="category-actions">
                      <button 
                        onClick={() => setEditingCategory(category.id)}
                        className="btn btn-small btn-secondary"
                      >
                        Edit
                      </button>
                      <button 
                        onClick={() => handleRemoveCategory(category.id)}
                        className="btn btn-small btn-danger"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const ComponentForm = ({ component, categories, onSave, onCancel }) => {
  const [formData, setFormData] = useState(component);

  const handleChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleConnectionsChange = (value) => {
    const connections = value.split(',').map(c => c.trim()).filter(c => c);
    setFormData(prev => ({
      ...prev,
      connections
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!formData.id || !formData.title) {
      alert('ID and Title are required');
      return;
    }
    onSave(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="component-form">
      <div className="form-row">
        <div className="form-group">
          <label>ID *</label>
          <input
            type="text"
            value={formData.id}
            onChange={(e) => handleChange('id', e.target.value)}
            required
          />
        </div>
        <div className="form-group">
          <label>Title *</label>
          <input
            type="text"
            value={formData.title}
            onChange={(e) => handleChange('title', e.target.value)}
            required
          />
        </div>
      </div>

      <div className="form-row">
        <div className="form-group">
          <label>Category</label>
          <select
            value={formData.category}
            onChange={(e) => handleChange('category', e.target.value)}
          >
            {Object.keys(categories).map(catId => (
              <option key={catId} value={catId}>{categories[catId].name}</option>
            ))}
          </select>
        </div>
        <div className="form-group">
          <label>Connections (comma-separated)</label>
          <input
            type="text"
            value={formData.connections?.join(', ') || ''}
            onChange={(e) => handleConnectionsChange(e.target.value)}
            placeholder="embeddings, vector-db, rag"
          />
        </div>
      </div>

      <div className="form-group">
        <label>Description</label>
        <textarea
          value={formData.description}
          onChange={(e) => handleChange('description', e.target.value)}
          rows="2"
        />
      </div>

      <div className="form-group">
        <label>Details (Markdown supported)</label>
        <textarea
          value={formData.details}
          onChange={(e) => handleChange('details', e.target.value)}
          rows="10"
        />
      </div>

      <div className="form-actions">
        <button type="submit" className="btn btn-primary">Save</button>
        <button type="button" onClick={onCancel} className="btn btn-secondary">Cancel</button>
      </div>
    </form>
  );
};

const CategoryForm = ({ category, onSave, onCancel }) => {
  const [formData, setFormData] = useState(category);

  const handleChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!formData.id || !formData.name) {
      alert('ID and Name are required');
      return;
    }
    onSave(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="category-form">
      <div className="form-row">
        <div className="form-group">
          <label>ID *</label>
          <input
            type="text"
            value={formData.id}
            onChange={(e) => handleChange('id', e.target.value)}
            required
          />
        </div>
        <div className="form-group">
          <label>Name *</label>
          <input
            type="text"
            value={formData.name}
            onChange={(e) => handleChange('name', e.target.value)}
            required
          />
        </div>
      </div>

      <div className="form-row">
        <div className="form-group">
          <label>Color</label>
          <input
            type="color"
            value={formData.color}
            onChange={(e) => handleChange('color', e.target.value)}
          />
        </div>
      </div>

      <div className="form-group">
        <label>Description</label>
        <textarea
          value={formData.description}
          onChange={(e) => handleChange('description', e.target.value)}
          rows="3"
        />
      </div>

      <div className="form-actions">
        <button type="submit" className="btn btn-primary">Save</button>
        <button type="button" onClick={onCancel} className="btn btn-secondary">Cancel</button>
      </div>
    </form>
  );
};

export default DataManager;
