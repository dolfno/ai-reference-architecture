/**
 * Storage utilities for managing architecture data in localStorage
 */

const STORAGE_KEYS = {
  ARCHITECTURE_COMPONENTS: 'ai-reference-architecture-components',
  CATEGORIES: 'ai-reference-architecture-categories'
};

// Default data that will be used if localStorage is empty
const defaultArchitectureComponents = {
  "llm-models": {
    id: "llm-models",
    title: "Large Language Models",
    category: "core",
    description: "Foundation models that power GenAI applications",
    details: `Large Language Models (LLMs) are neural networks trained on vast amounts of text data. They form the foundation of most GenAI applications.

**Key Characteristics:**
- Pre-trained on diverse text corpora
- Capable of understanding and generating human-like text
- Can perform various tasks through prompting
- Available in different sizes (parameters ranging from millions to trillions)

**Common Models:**
- OpenAI GPT-4, GPT-3.5
- Anthropic Claude
- Google Gemini, PaLM
- Meta LLaMA
- Mistral AI models

**Use Cases:**
- Text generation and completion
- Question answering
- Summarization
- Translation
- Code generation`,
    connections: ["embeddings", "prompt-engineering", "fine-tuning"]
  },

  "embeddings": {
    id: "embeddings",
    title: "Embeddings",
    category: "core",
    description: "Vector representations of text for semantic understanding",
    details: `Embeddings convert text into high-dimensional vectors that capture semantic meaning, enabling machines to understand relationships between words and concepts.

**How They Work:**
- Text is converted to numerical vectors (typically 384-3072 dimensions)
- Similar concepts have vectors close together in vector space
- Enable semantic search and similarity matching

**Popular Models:**
- OpenAI text-embedding-ada-002, text-embedding-3
- Sentence Transformers (BERT-based)
- Cohere embeddings
- Google's Universal Sentence Encoder

**Applications:**
- Semantic search
- Document similarity
- Clustering and classification
- Recommendation systems
- RAG (Retrieval Augmented Generation)`,
    connections: ["vector-db", "rag", "semantic-search"]
  },

  "vector-db": {
    id: "vector-db",
    title: "Vector Databases",
    category: "core",
    description: "Specialized databases for storing and querying embeddings",
    details: `Vector databases are optimized for storing, indexing, and querying high-dimensional vectors (embeddings) efficiently.

**Key Features:**
- Fast similarity search (ANN - Approximate Nearest Neighbor)
- Horizontal scalability
- Metadata filtering
- Hybrid search (vector + keyword)

**Popular Solutions:**
- Pinecone
- Weaviate
- Qdrant
- Chroma
- Milvus
- pgvector (PostgreSQL extension)

**Use Cases:**
- Semantic search engines
- RAG knowledge bases
- Recommendation systems
- Duplicate detection
- Anomaly detection`,
    connections: ["embeddings", "rag", "semantic-search"]
  },

  "prompt-engineering": {
    id: "prompt-engineering",
    title: "Prompt Engineering",
    category: "core",
    description: "Techniques for crafting effective prompts to guide LLM behavior",
    details: `Prompt engineering is the practice of designing inputs to elicit desired outputs from language models.

**Core Techniques:**

1. **Zero-shot prompting**: Direct instruction without examples
2. **Few-shot prompting**: Providing examples in the prompt
3. **Chain-of-Thought (CoT)**: Encouraging step-by-step reasoning
4. **Role prompting**: Assigning a role or persona
5. **Template prompting**: Using structured formats

**Best Practices:**
- Be specific and clear
- Provide context and constraints
- Use delimiters to separate sections
- Iterate and test variations
- Include output format specifications

**Advanced Patterns:**
- System/User/Assistant message roles
- Constitutional AI principles
- Self-consistency prompting
- ReAct (Reasoning + Acting)`,
    connections: ["llm-models", "agents", "guardrails"]
  },

  "rag": {
    id: "rag",
    title: "RAG (Retrieval Augmented Generation)",
    category: "advanced",
    description: "Combining retrieval with generation for knowledge-grounded responses",
    details: `RAG enhances LLM responses by retrieving relevant information from external knowledge bases before generation.

**Architecture:**
1. **Query Processing**: Convert user query to embedding
2. **Retrieval**: Search vector database for relevant documents
3. **Context Assembly**: Combine retrieved docs with query
4. **Generation**: LLM generates response using context
5. **Post-processing**: Format and validate output

**Benefits:**
- Reduces hallucinations
- Provides up-to-date information
- Enables source attribution
- Reduces need for fine-tuning
- Domain-specific knowledge without retraining

**Variants:**
- Simple RAG: Basic retrieval + generation
- Conversational RAG: With chat history
- Multi-query RAG: Multiple retrieval strategies
- Agentic RAG: With reasoning and tool use
- GraphRAG: Using knowledge graphs`,
    connections: ["vector-db", "embeddings", "llm-models", "agents"]
  },

  "agents": {
    id: "agents",
    title: "AI Agents",
    category: "advanced",
    description: "Autonomous systems that can plan, reason, and take actions",
    details: `AI Agents are systems that use LLMs to autonomously complete complex tasks through planning, reasoning, and tool use.

**Core Capabilities:**
- **Planning**: Breaking down tasks into steps
- **Memory**: Maintaining context and history
- **Tool Use**: Calling external APIs and functions
- **Reflection**: Self-evaluation and error correction
- **Multi-step Reasoning**: Chaining thoughts and actions

**Frameworks:**
- LangChain Agents
- AutoGPT / AgentGPT
- BabyAGI
- CrewAI
- Semantic Kernel
- OpenAI Assistants API

**Agent Patterns:**
- ReAct (Reason + Act)
- Plan-and-Execute
- Reflexion
- Multi-agent collaboration
- Tool-calling agents

**Use Cases:**
- Research assistants
- Data analysis automation
- Code generation and debugging
- Customer support automation`,
    connections: ["llm-models", "rag", "function-calling", "orchestration"]
  },

  "function-calling": {
    id: "function-calling",
    title: "Function Calling",
    category: "advanced",
    description: "Enabling LLMs to call external tools and APIs",
    details: `Function calling allows LLMs to interact with external systems by invoking predefined functions with structured parameters.

**How It Works:**
1. Define available functions with schemas
2. LLM decides which function to call
3. LLM generates structured arguments
4. Application executes the function
5. Result is returned to LLM
6. LLM incorporates result in response

**Supported Platforms:**
- OpenAI Functions
- Anthropic Tool Use
- Google Function Calling
- LangChain Tools
- Semantic Kernel Plugins

**Common Use Cases:**
- Database queries
- API integrations
- Calculator operations
- Web searches
- File operations
- Email sending

**Best Practices:**
- Clear function descriptions
- Type-safe parameter schemas
- Error handling
- Rate limiting
- Security validation`,
    connections: ["agents", "llm-models", "orchestration"]
  },

  "fine-tuning": {
    id: "fine-tuning",
    title: "Fine-tuning",
    category: "advanced",
    description: "Customizing models for specific tasks or domains",
    details: `Fine-tuning adapts pre-trained models to specific tasks or domains by training on custom datasets.

**Approaches:**

1. **Full Fine-tuning**: Update all model parameters
2. **PEFT (Parameter-Efficient Fine-Tuning)**:
   - LoRA (Low-Rank Adaptation)
   - QLoRA (Quantized LoRA)
   - Adapter layers
   - Prefix tuning

**When to Fine-tune:**
- Need consistent output format
- Domain-specific terminology
- Behavior modification
- Performance optimization
- Reduce prompting complexity

**Considerations:**
- Data requirements (100s-10000s examples)
- Cost and compute resources
- Ongoing maintenance
- Evaluation metrics
- Version control

**Alternatives:**
- Few-shot prompting
- RAG for knowledge
- Prompt engineering
- Instruction tuning`,
    connections: ["llm-models", "evaluation", "data-pipeline"]
  },

  "evaluation": {
    id: "evaluation",
    title: "Evaluation & Testing",
    category: "advanced",
    description: "Measuring and validating GenAI application quality",
    details: `Systematic evaluation ensures GenAI applications meet quality, safety, and performance requirements.

**Evaluation Dimensions:**

1. **Quality Metrics**:
   - Accuracy and correctness
   - Relevance and coherence
   - Completeness
   - Hallucination rate

2. **LLM-as-Judge**:
   - Using LLMs to evaluate outputs
   - Comparative evaluation
   - Rubric-based scoring

3. **Human Evaluation**:
   - Expert review
   - User feedback
   - A/B testing

4. **Automated Metrics**:
   - BLEU, ROUGE (text similarity)
   - Perplexity
   - Embedding similarity
   - Custom heuristics

**Tools:**
- LangSmith
- Arize Phoenix
- Weights & Biases
- MLflow
- PromptLayer
- Humanloop

**Best Practices:**
- Create diverse test sets
- Track prompt versions
- Monitor in production
- Regression testing`,
    connections: ["llm-models", "monitoring", "guardrails"]
  },

  "guardrails": {
    id: "guardrails",
    title: "Guardrails",
    category: "advanced",
    description: "Safety and validation mechanisms for LLM outputs",
    details: `Guardrails are systems that ensure LLM inputs and outputs meet safety, quality, and business requirements.

**Types of Guardrails:**

1. **Input Validation**:
   - Prompt injection detection
   - PII detection and redaction
   - Content filtering
   - Rate limiting

2. **Output Validation**:
   - Toxicity detection
   - Hallucination checking
   - Format validation
   - Fact verification
   - Bias detection

3. **Business Rules**:
   - Length constraints
   - Required elements
   - Prohibited content
   - Compliance requirements

**Implementation Approaches:**
- Rule-based filters
- Classifier models
- LLM-based validation
- Structured output enforcement

**Tools:**
- Guardrails AI
- NeMo Guardrails (NVIDIA)
- LangKit
- Llama Guard
- Azure AI Content Safety`,
    connections: ["llm-models", "evaluation", "monitoring"]
  },

  "semantic-search": {
    id: "semantic-search",
    title: "Semantic Search",
    category: "advanced",
    description: "Meaning-based search using embeddings",
    details: `Semantic search finds information based on meaning and intent rather than just keyword matching.

**How It Works:**
1. Documents are converted to embeddings
2. User query is embedded
3. Vector similarity search finds relevant docs
4. Results ranked by semantic relevance

**Advantages Over Keyword Search:**
- Understands synonyms and related concepts
- Handles paraphrasing
- Cross-lingual capabilities
- Better handles ambiguity
- Intent understanding

**Hybrid Search:**
Combining semantic and keyword search:
- Semantic for meaning
- Keyword (BM25) for exact matches
- Weighted fusion of results

**Applications:**
- Document search engines
- FAQ matching
- Product discovery
- Code search
- Research paper retrieval

**Implementation:**
- Embed all documents offline
- Store in vector database
- Embed queries in real-time
- Return top-K similar documents`,
    connections: ["embeddings", "vector-db", "rag"]
  },

  "orchestration": {
    id: "orchestration",
    title: "Orchestration",
    category: "advanced",
    description: "Managing complex workflows and multi-step processes",
    details: `Orchestration coordinates multiple components and steps in GenAI applications to create complex workflows.

**Key Concepts:**

1. **Workflow Management**:
   - Sequential chains
   - Parallel execution
   - Conditional branching
   - Error handling
   - Retry logic

2. **State Management**:
   - Conversation history
   - Intermediate results
   - Context passing
   - Memory systems

3. **Component Integration**:
   - LLM calls
   - Vector database queries
   - Tool invocations
   - API requests

**Orchestration Frameworks:**
- LangChain (LCEL - LangChain Expression Language)
- LlamaIndex
- Semantic Kernel
- Haystack
- Temporal (for production workflows)

**Patterns:**
- Sequential chains
- Map-reduce
- Router chains
- Multi-agent orchestration
- Feedback loops

**Use Cases:**
- Complex agent workflows
- Multi-step data processing
- Conversational AI
- Document processing pipelines`,
    connections: ["agents", "rag", "function-calling"]
  },

  "data-pipeline": {
    id: "data-pipeline",
    title: "Data Pipeline",
    category: "infrastructure",
    description: "Processing and preparing data for GenAI applications",
    details: `Data pipelines handle the collection, processing, and transformation of data for GenAI systems.

**Pipeline Stages:**

1. **Data Ingestion**:
   - Document loaders
   - API connectors
   - Web scraping
   - File parsing (PDF, HTML, etc.)

2. **Processing**:
   - Chunking strategies
   - Cleaning and normalization
   - Metadata extraction
   - Deduplication

3. **Transformation**:
   - Format conversion
   - Enrichment
   - Summarization
   - Entity extraction

4. **Loading**:
   - Embedding generation
   - Vector database indexing
   - Metadata storage

**Chunking Strategies:**
- Fixed-size chunks
- Semantic chunking
- Document structure-aware
- Overlapping windows
- Hierarchical chunking

**Tools:**
- Apache Airflow
- Unstructured.io
- LangChain document loaders
- LlamaIndex data connectors`,
    connections: ["embeddings", "vector-db", "fine-tuning"]
  },

  "monitoring": {
    id: "monitoring",
    title: "Monitoring & Observability",
    category: "infrastructure",
    description: "Tracking performance, quality, and costs in production",
    details: `Monitoring provides visibility into GenAI application behavior, quality, and resource usage in production.

**Key Metrics:**

1. **Performance**:
   - Latency (TTFB, total)
   - Throughput
   - Token usage
   - Error rates

2. **Quality**:
   - User feedback
   - Hallucination detection
   - Response relevance
   - Task completion rate

3. **Cost**:
   - Token consumption
   - API costs
   - Infrastructure costs
   - Cost per interaction

4. **Usage**:
   - Request volume
   - User engagement
   - Feature adoption
   - Error patterns

**Monitoring Tools:**
- LangSmith
- Arize AI
- Weights & Biases
- DataDog
- Helicone
- LangFuse

**Best Practices:**
- Log all LLM interactions
- Track prompt versions
- Monitor for drift
- Set up alerts
- A/B test changes`,
    connections: ["evaluation", "guardrails", "deployment"]
  },

  "deployment": {
    id: "deployment",
    title: "Deployment",
    category: "infrastructure",
    description: "Strategies for deploying GenAI applications to production",
    details: `Deployment involves making GenAI applications available to users reliably and efficiently.

**Deployment Options:**

1. **API-Based**:
   - OpenAI, Anthropic, Google APIs
   - Pay-per-use pricing
   - No infrastructure management
   - Quick to deploy

2. **Self-Hosted**:
   - Open-source models
   - Full control
   - Higher upfront cost
   - Privacy benefits

3. **Hybrid**:
   - API for LLM
   - Self-hosted supporting services
   - Balanced approach

**Infrastructure Considerations:**

- **Containerization**: Docker, Kubernetes
- **Serverless**: AWS Lambda, Cloud Functions
- **GPU Requirements**: For local models
- **Auto-scaling**: Handle variable load
- **Rate Limiting**: Protect against abuse
- **Caching**: Reduce costs and latency

**Deployment Platforms:**
- Vercel, Netlify (web apps)
- AWS, GCP, Azure (cloud)
- Hugging Face Spaces
- Modal, Replicate
- RunPod, Lambda Labs (GPU)`,
    connections: ["monitoring", "scaling", "security"]
  },

  "scaling": {
    id: "scaling",
    title: "Scaling",
    category: "infrastructure",
    description: "Handling increased load and optimizing performance",
    details: `Scaling strategies ensure GenAI applications perform well under varying loads while managing costs.

**Scaling Dimensions:**

1. **Request Volume**:
   - Load balancing
   - Queue management
   - Rate limiting
   - Caching strategies

2. **Response Time**:
   - Prompt optimization
   - Model selection
   - Streaming responses
   - Parallel processing

3. **Cost**:
   - Token optimization
   - Caching frequent queries
   - Batching requests
   - Model routing (cheap vs expensive)

**Optimization Techniques:**

- **Caching**:
  - Semantic caching (similar queries)
  - Exact match caching
  - Response memoization

- **Batching**:
  - Combine multiple requests
  - Parallel processing

- **Model Routing**:
  - Use smaller models when possible
  - Cascade: try cheap model first
  - Task-specific model selection

**Architecture Patterns:**
- Horizontal scaling
- Async processing
- CDN for static assets
- Database read replicas`,
    connections: ["deployment", "monitoring", "caching"]
  },

  "caching": {
    id: "caching",
    title: "Caching",
    category: "infrastructure",
    description: "Storing and reusing results to reduce costs and latency",
    details: `Caching in GenAI applications reduces costs and improves response times by reusing previous results.

**Caching Strategies:**

1. **Exact Match Caching**:
   - Cache identical prompts
   - Fast lookups
   - Limited applicability

2. **Semantic Caching**:
   - Cache similar queries
   - Use embedding similarity
   - More flexible matching
   - Configurable similarity threshold

3. **Partial Caching**:
   - Cache prompt components
   - Anthropic prompt caching
   - Reduce repeated context tokens

**Cache Layers:**

- **Application Level**:
  - In-memory (Redis, Memcached)
  - Fast access
  - Response caching

- **LLM Provider Level**:
  - Prompt caching (Anthropic)
  - Reduced token costs
  - Lower latency

- **Vector DB Level**:
  - Query result caching
  - Embedding caching

**Best Practices:**
- Set appropriate TTLs
- Monitor hit rates
- Invalidate stale data
- Cache at multiple levels
- Consider privacy implications`,
    connections: ["scaling", "deployment", "embeddings"]
  },

  "security": {
    id: "security",
    title: "Security",
    category: "infrastructure",
    description: "Protecting GenAI applications from threats and vulnerabilities",
    details: `Security measures protect GenAI applications from attacks, data breaches, and misuse.

**Threat Vectors:**

1. **Prompt Injection**:
   - Malicious prompt crafting
   - Jailbreaking attempts
   - Instruction override
   - Data exfiltration

2. **Data Privacy**:
   - PII exposure
   - Training data leakage
   - Unintended memorization

3. **Resource Abuse**:
   - DoS attacks
   - Cost exploitation
   - Rate limit evasion

**Security Measures:**

1. **Input Validation**:
   - Prompt injection detection
   - Input sanitization
   - PII redaction
   - Content filtering

2. **Access Control**:
   - Authentication
   - Authorization
   - API key management
   - Role-based access

3. **Output Validation**:
   - Content filtering
   - Data loss prevention
   - Sensitive info detection

4. **Infrastructure**:
   - Encryption (in transit & at rest)
   - Network security
   - Audit logging
   - Secure key storage

**Tools:**
- OWASP LLM Top 10
- Prompt armor/shields
- DLP solutions
- WAF (Web Application Firewall)`,
    connections: ["guardrails", "deployment", "monitoring"]
  }
};

const defaultCategories = {
  core: {
    id: "core",
    name: "Core Components",
    color: "#3B82F6",
    description: "Fundamental building blocks of GenAI applications"
  },
  advanced: {
    id: "advanced",
    name: "Advanced Patterns",
    color: "#8B5CF6",
    description: "Sophisticated techniques for complex use cases"
  },
  infrastructure: {
    id: "infrastructure",
    name: "Infrastructure & Operations",
    color: "#10B981",
    description: "Production deployment and operational concerns"
  }
};

/**
 * Load architecture components from localStorage
 * @returns {Object} Architecture components object
 */
export function loadArchitectureComponents() {
  try {
    const stored = localStorage.getItem(STORAGE_KEYS.ARCHITECTURE_COMPONENTS);
    if (stored) {
      return JSON.parse(stored);
    }
  } catch (error) {
    console.warn('Failed to load architecture components from localStorage:', error);
  }
  
  // Return default data if nothing in localStorage or error occurred
  return defaultArchitectureComponents;
}

/**
 * Save architecture components to localStorage
 * @param {Object} components - Architecture components object
 */
export function saveArchitectureComponents(components) {
  try {
    localStorage.setItem(STORAGE_KEYS.ARCHITECTURE_COMPONENTS, JSON.stringify(components));
    return true;
  } catch (error) {
    console.error('Failed to save architecture components to localStorage:', error);
    return false;
  }
}

/**
 * Load categories from localStorage
 * @returns {Object} Categories object
 */
export function loadCategories() {
  try {
    const stored = localStorage.getItem(STORAGE_KEYS.CATEGORIES);
    if (stored) {
      return JSON.parse(stored);
    }
  } catch (error) {
    console.warn('Failed to load categories from localStorage:', error);
  }
  
  // Return default data if nothing in localStorage or error occurred
  return defaultCategories;
}

/**
 * Save categories to localStorage
 * @param {Object} categories - Categories object
 */
export function saveCategories(categories) {
  try {
    localStorage.setItem(STORAGE_KEYS.CATEGORIES, JSON.stringify(categories));
    return true;
  } catch (error) {
    console.error('Failed to save categories to localStorage:', error);
    return false;
  }
}

/**
 * Add or update a single component
 * @param {Object} component - Component object with id, title, category, description, details, connections
 */
export function updateComponent(component) {
  const components = loadArchitectureComponents();
  components[component.id] = component;
  return saveArchitectureComponents(components);
}

/**
 * Remove a component by ID
 * @param {string} componentId - ID of the component to remove
 */
export function removeComponent(componentId) {
  const components = loadArchitectureComponents();
  delete components[componentId];
  return saveArchitectureComponents(components);
}

/**
 * Add or update a category
 * @param {Object} category - Category object with id, name, color, description
 */
export function updateCategory(category) {
  const categories = loadCategories();
  categories[category.id] = category;
  return saveCategories(categories);
}

/**
 * Remove a category by ID
 * @param {string} categoryId - ID of the category to remove
 */
export function removeCategory(categoryId) {
  const categories = loadCategories();
  delete categories[categoryId];
  return saveCategories(categories);
}

/**
 * Initialize localStorage with default data if it's empty
 */
export function initializeStorage() {
  // Check if localStorage is empty and initialize with defaults
  if (!localStorage.getItem(STORAGE_KEYS.ARCHITECTURE_COMPONENTS)) {
    saveArchitectureComponents(defaultArchitectureComponents);
  }
  
  if (!localStorage.getItem(STORAGE_KEYS.CATEGORIES)) {
    saveCategories(defaultCategories);
  }
}

/**
 * Reset storage to default values
 */
export function resetToDefaults() {
  saveArchitectureComponents(defaultArchitectureComponents);
  saveCategories(defaultCategories);
}

/**
 * Export all data for backup
 * @returns {Object} Complete data export
 */
export function exportData() {
  return {
    components: loadArchitectureComponents(),
    categories: loadCategories(),
    exportDate: new Date().toISOString()
  };
}

/**
 * Import data from backup
 * @param {Object} data - Data object with components and categories
 */
export function importData(data) {
  try {
    if (data.components) {
      saveArchitectureComponents(data.components);
    }
    if (data.categories) {
      saveCategories(data.categories);
    }
    return true;
  } catch (error) {
    console.error('Failed to import data:', error);
    return false;
  }
}
