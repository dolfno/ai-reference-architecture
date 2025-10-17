import {
   loadArchitectureComponents,
   loadCategories,
   initializeStorage
} from '../utils/storageUtils.js';

// Initialize storage with default data if empty
initializeStorage();

// Load architecture components dynamically from localStorage
export const architectureComponents = loadArchitectureComponents();

// Load categories dynamically from localStorage
export const categories = loadCategories();
