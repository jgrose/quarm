# PROJECT PLAN: Complex Test Plan

## Objective
Build a full-stack inventory management system with frontend, backend, security, and documentation.

## Sub-Agents

### AGENT: backend_engineer
- description: Python/FastAPI backend engineer. Builds secure REST APIs and data pipelines.
- tools: execute_code, write_file, read_file

### AGENT: frontend_engineer
- description: React/TypeScript frontend engineer. Builds accessible, responsive UI components.
- tools: write_file, design_ui

### AGENT: technical_writer
- description: Technical writer specialising in developer and user-facing documentation.
- tools: write_file, read_file

## Managers

### MANAGER: engineering_director
- title: Engineering Architecture Director
- description: Senior engineering leader who reviews backend code and security architecture.
- expertise_blend: [API_design, Python_architecture, cloud, security]
- oversees: [backend_engineer]

### MANAGER: product_director
- title: Product and Delivery Director
- description: Product leader who reviews user-facing outputs for alignment with user needs.
- expertise_blend: [product_management, UX_strategy, technical_communication]
- oversees: [frontend_engineer, technical_writer]

## Custom Reviewers

### REVIEWER: api_standards_reviewer
- title: API Standards Reviewer
- description: Reviews API designs for RESTful best practices, consistent naming, proper status codes, and versioning.
- focus_areas: [REST conventions, naming consistency, status codes, versioning, pagination]
- applies_to: [code, api, backend]
- tolerance: 5

## Tasks

### TASK-001
- title: Design database schema
- agent: backend_engineer
- description: Design the PostgreSQL schema for inventory items, categories, suppliers, and stock levels.
- task_type: [code, backend, data]
- reviewers: [security_engineer]
- depends_on: []
- tolerance: 7

### TASK-002
- title: Build inventory CRUD API
- agent: backend_engineer
- description: Build FastAPI endpoints for creating, reading, updating, and deleting inventory items.
- task_type: [code, api, backend]
- reviewers: [security_engineer, api_standards_reviewer]
- depends_on: [TASK-001]

### TASK-003
- title: Build inventory dashboard UI
- agent: frontend_engineer
- description: Build a React dashboard showing inventory levels, low-stock alerts, and category breakdowns.
- task_type: [code, ui, frontend, dashboard]
- reviewers: [ux_designer, user_tester]
- depends_on: [TASK-002]

### TASK-004
- title: Write API and user documentation
- agent: technical_writer
- description: Write developer API reference and end-user guide for the inventory management system.
- task_type: [documentation, report, user_flow]
- reviewers: [user_tester]
- depends_on: [TASK-002, TASK-003]
