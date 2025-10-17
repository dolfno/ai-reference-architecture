# GenAI Architecture Reference

An interactive web application that helps developers understand the components, patterns, and infrastructure of Generative AI applications.

![GenAI Architecture Reference](https://img.shields.io/badge/Status-Active-success)
![React](https://img.shields.io/badge/React-18.3-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This application provides an interactive, visual reference guide to GenAI architecture. It covers:

- **Core Components**: LLMs, embeddings, vector databases, and prompt engineering
- **Advanced Patterns**: RAG, AI agents, fine-tuning, evaluation, and guardrails
- **Infrastructure**: Deployment, monitoring, scaling, security, and caching

## Features

- **Interactive Architecture Diagram**: Explore components and their relationships visually using ReactFlow
- **Detailed Component Information**: Click any component to view comprehensive explanations, use cases, and best practices
- **Connected Learning**: Navigate between related components to understand how they work together
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Beautiful UI**: Modern, gradient-based design with smooth animations

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
│   │   └── ComponentDetail.css
│   ├── data/
│   │   └── architectureData.js         # All component data and relationships
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

### Advanced Patterns
- **RAG (Retrieval Augmented Generation)**: Knowledge-grounded responses
- **AI Agents**: Autonomous systems with planning and reasoning
- **Function Calling**: LLM integration with external tools
- **Fine-tuning**: Model customization
- **Evaluation & Testing**: Quality assurance
- **Guardrails**: Safety and validation
- **Semantic Search**: Meaning-based search
- **Orchestration**: Workflow management

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

### Adding New Components

Edit `src/data/architectureData.js` to add new components:

```javascript
"new-component": {
  id: "new-component",
  title: "New Component",
  category: "core", // or "advanced" or "infrastructure"
  description: "Brief description",
  details: `Detailed information with markdown-like formatting...`,
  connections: ["related-component-1", "related-component-2"]
}
```

### Styling

- Global styles: `src/index.css`
- App layout: `src/App.css`
- Diagram styles: `src/components/ArchitectureDiagram.css`
- Detail panel: `src/components/ComponentDetail.css`

### Colors

Category colors are defined in `src/data/architectureData.js`:
- Core Components: `#3B82F6` (blue)
- Advanced Patterns: `#8B5CF6` (purple)
- Infrastructure: `#10B981` (green)

## Use Cases

This reference is useful for:

- **Developers** building GenAI applications
- **Architects** designing GenAI systems
- **Product Managers** understanding capabilities
- **Students** learning GenAI concepts
- **Teams** aligning on architecture decisions

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Areas for contribution:
- Adding more components
- Improving component descriptions
- Adding code examples
- Enhancing visualizations
- Fixing bugs

## Future Enhancements

Potential improvements:
- [ ] Code examples for each component
- [ ] Search functionality
- [ ] Filter by category
- [ ] Export diagram as image
- [ ] Dark mode
- [ ] Real-world architecture patterns
- [ ] Integration examples
- [ ] Video tutorials
- [ ] Cost calculators
- [ ] Community-contributed patterns

## License

MIT License - feel free to use this project for learning and commercial purposes.

## Resources

- [OpenAI Documentation](https://platform.openai.com/docs)
- [Anthropic Claude Docs](https://docs.anthropic.com)
- [LangChain Documentation](https://python.langchain.com)
- [Hugging Face](https://huggingface.co)
- [ReactFlow Documentation](https://reactflow.dev)

## Support

If you find this helpful, please star the repository and share it with others building GenAI applications!

For issues or questions, please open a GitHub issue.

---

Built with ❤️ to help developers understand GenAI architectures
