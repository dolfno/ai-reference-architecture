# GenAI Reference Architecture

An interactive web application that helps developers understand the components, patterns, and infrastructure of Generative AI applications.

![GenAI Reference Architecture](https://img.shields.io/badge/Status-Active-success)
![React](https://img.shields.io/badge/React-18.3-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This application provides an interactive, visual reference guide to GenAI architecture. It covers:

- **Core Components**: LLMs, embeddings, vector databases, and prompt engineering
- **Advanced Patterns**: RAG, AI agents, fine-tuning, evaluation, and guardrails
- **Infrastructure**: Deployment, monitoring, scaling, security, and caching

## Screenshots

### Interactive Architecture Diagram
![Architecture Diagram](./screenshots/architecture-diagram.png)
*Explore the complete GenAI architecture with interactive components and connections*

### Component Details Panel
![Component Details](./screenshots/component-details.png)
*Detailed explanations, use cases, and best practices for each component*

## Features

- **Interactive Architecture Diagram**: Explore components and their relationships visually using ReactFlow
- **Detailed Component Information**: Click any component to view comprehensive explanations, use cases, and best practices
- **Connected Learning**: Navigate between related components to understand how they work together
- **Dynamic Data Management**: Add, edit, and customize components through the built-in data manager
- **Local Storage Persistence**: All customizations are saved locally in your browser
- **Data Import/Export**: Backup and share your custom architecture data
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Beautiful UI**: Modern, gradient-based design with smooth animations

## Use Cases & User Stories

### Use Case 1: Learning & Education

#### Business User Stories
- **As a product manager**, I want to explore the high-level AI architecture overview so I can understand how different components impact business outcomes and user experience
- **As a business stakeholder**, I want to see how components connect and work together so I can identify potential bottlenecks and optimization opportunities
- **As a project manager**, I want to understand the relationships between AI components so I can better scope projects and allocate resources
- **As a non-technical executive**, I want to grasp the purpose and value of each AI component so I can make informed strategic decisions

#### Technical User Stories
- **As a software engineer**, I want to drill deep into specific components so I can understand implementation details and integration patterns
- **As a data scientist**, I want to explore technical specifications and parameters of AI models so I can optimize performance for my use case
- **As a DevOps engineer**, I want to understand infrastructure requirements and scaling considerations so I can design robust deployment strategies
- **As an AI researcher**, I want to access detailed algorithmic explanations and performance characteristics so I can evaluate components for my research

#### Learning Navigation Stories
- **As a learner**, I want to follow connected learning paths between related components so I can build comprehensive understanding
- **As a business user**, I want to stay at the overview level and see system-wide flows so I can understand the big picture without getting overwhelmed by technical details
- **As a technical expert**, I want quick access to deep-dive sections and implementation examples so I can efficiently find the specific information I need

### Use Case 2: Architecture Design & Planning

#### Architect/Designer User Stories
- **As a solution architect**, I want to drag and drop components to design a custom AI architecture so I can visualize the solution before implementation
- **As a technical lead**, I want to map out the flow and connections between components for a specific use case (e.g., chatbot, content generation, document analysis) so I can create implementation blueprints
- **As an AI engineer**, I want to select appropriate components based on my use case requirements so I can build an optimal architecture
- **As a system designer**, I want to validate that my component connections make technical sense so I can avoid architectural mistakes early

#### Use Case-Specific Design Stories
- **As a chatbot architect**, I want to design the conversation flow from user input through NLP processing, knowledge retrieval, and response generation so I can plan the complete system
- **As a RAG system designer**, I want to configure the pipeline from document ingestion through chunking, embedding, storage, retrieval, and generation so I can optimize for my specific content type
- **As a content generation architect**, I want to design workflows with prompt engineering, fine-tuned models, and output structuring so I can deliver consistent, high-quality content

#### Planning & Validation Stories
- **As an architect**, I want to export my custom architecture design so I can share it with development teams and stakeholders
- **As a project planner**, I want to estimate complexity and resource requirements based on my component selection so I can create realistic project timelines
- **As a technical reviewer**, I want to validate architecture designs against best practices and common patterns so I can provide informed feedback

## Getting Started

### Prerequisites

- Node.js 16+ and npm

### Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd ai-reference-architecture
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser to `http://localhost:3000`

### Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory, ready to deploy to any static hosting service.

## Project Structure

```
ai-reference-architecture/
├── src/
│   ├── components/
│   │   ├── ArchitectureDiagram.jsx    # Interactive diagram component
│   │   ├── ArchitectureDiagram.css
│   │   ├── ComponentDetail.jsx         # Detail panel for components
│   │   ├── ComponentDetail.css
│   │   ├── DataManager.jsx            # Data management interface
│   │   └── DataManager.css
│   ├── data/
│   │   └── architectureData.js         # Dynamic data loading
│   ├── utils/
│   │   └── storageUtils.js             # LocalStorage utilities
│   ├── App.jsx                          # Main application component
│   ├── App.css
│   ├── main.jsx                         # Entry point
│   └── index.css                        # Global styles
├── index.html
├── package.json
├── vite.config.js
└── README.md
```

## Components Covered

### Core Components
- **Large Language Models**: Foundation models (GPT-4, Claude, Gemini, etc.)
- **Embeddings**: Vector representations for semantic understanding
- **Vector Databases**: Specialized databases for embeddings (Pinecone, Weaviate, etc.)
- **Prompt Engineering**: Techniques for effective prompting
- **Reranker Model**: Models that reorder search results to improve relevance
- **Chunker**: Text splitting strategies for optimal processing and retrieval
- **Output Structurer**: Tools for formatting and structuring LLM outputs
- **Keyword Search**: Traditional text search using BM25, TF-IDF, trigrams

### Advanced Patterns
- **RAG (Retrieval Augmented Generation)**: Knowledge-grounded responses
- **AI Agents**: Autonomous systems with planning and reasoning
- **Function Calling**: LLM integration with external tools
- **Fine-tuning**: Model customization
- **Evaluation & Testing**: Quality assurance
- **Guardrails**: Safety and validation
- **Semantic Search**: Meaning-based search
- **Orchestration**: Workflow management
- **Community Detection**: Graph algorithms for identifying clusters of related entities
- **Relationship Extraction**: NLP techniques for identifying relationships between entities
- **Entity Extraction**: NLP technique for identifying and classifying named entities
- **Query Routing**: Directing queries to appropriate models or systems based on intent
- **LLM Graph Creator**: Tools for creating knowledge graphs using Large Language Models
- **Knowledge Graph**: Graph-based knowledge representation connecting entities and relationships
- **Sliding Context Window**: Dynamic context management for long conversations and documents

### Infrastructure
- **Data Pipeline**: Data processing and preparation
- **Monitoring & Observability**: Production tracking
- **Deployment**: Deployment strategies
- **Scaling**: Performance optimization
- **Caching**: Cost and latency reduction
- **Security**: Threat protection

## Technology Stack

- **React 18**: UI framework
- **ReactFlow**: Interactive diagram visualization
- **Vite**: Build tool and dev server
- **Lucide React**: Icon library
- **React Router**: Navigation (if needed for future enhancements)

## Customization

### Dynamic Data Management

The application now includes a built-in data manager accessible via the "Manage Data" button in the header. This allows you to:

- **Add New Components**: Create custom architecture components
- **Edit Existing Components**: Modify titles, descriptions, details, and connections
- **Delete Components**: Remove components you don't need
- **Manage Categories**: Create and customize component categories
- **Import/Export Data**: Backup and share your custom architecture data
- **Reset to Defaults**: Restore the original component set

All changes are automatically saved to localStorage and persist between browser sessions.

### Manual Data Editing

You can still manually edit the default data in `src/utils/storageUtils.js` by modifying the `defaultArchitectureComponents` and `defaultCategories` objects.

### Adding Components Programmatically

Use the storage utilities in your code:

```javascript
import { updateComponent, loadArchitectureComponents } from './src/utils/storageUtils.js';

// Add a new component
updateComponent({
  id: "new-component",
  title: "New Component",
  category: "core",
  description: "Brief description",
  details: `Detailed information...`,
  connections: ["related-component-1"]
});
```

## License

MIT License - feel free to use this project for learning and commercial purposes.

## Support

If you find this helpful, please star the repository and share it with others building GenAI applications!

For issues or questions, please open a GitHub issue.

---

Built with ❤️ to help developers understand GenAI architectures
