# PROJECT PLAN: AWS Cost Intelligence Dashboard

## Objective
Build a web dashboard that ingests AWS Cost Explorer data, surfaces cost anomalies and trends by service and team, and provides actionable recommendations — designed for both engineers and non-technical stakeholders.

## Sub-Agents

### AGENT: backend_engineer
- description: Python/FastAPI backend engineer. Builds secure REST APIs, data ingestion pipelines, and business logic. Produces production-ready Python code with type hints, error handling, logging, and OpenAPI documentation. Knows AWS SDK (boto3), cost allocation tags, and data normalization patterns.
- tools: execute_code, write_file, read_file

### AGENT: security_architect
- description: Application security specialist. Designs auth flows, IAM policies, secrets management patterns, and data access controls. Produces architecture decision records (ADRs), security configuration files, and threat model documents. Applies least-privilege, zero-trust, and OWASP best practices by default.
- tools: write_file, reason

### AGENT: frontend_engineer
- description: React/TypeScript frontend engineer. Builds accessible, responsive UI components, data visualizations, and user flows. Produces JSX components, Tailwind CSS styling, and chart configurations using Recharts or D3. Prioritises accessibility (WCAG 2.1 AA), keyboard nav, and clear information hierarchy.
- tools: write_file, design_ui

### AGENT: technical_writer
- description: Technical writer specialising in developer and user-facing documentation. Produces clear READMEs, API references, onboarding guides, and in-app help content. Writes for two audiences simultaneously: engineers needing implementation detail and stakeholders needing plain-English explanations.
- tools: write_file, read_file

## Managers

### MANAGER: engineering_director
- title: Engineering Architecture Director
- description: Senior engineering leader who reviews backend code and security architecture for correctness, scalability, and adherence to cloud-native best practices. Evaluates whether APIs are well-designed, data models are appropriate, and security posture is production-ready.
- expertise_blend: [API_design, Python_architecture, AWS_cloud, data_modeling, secure_coding, performance_optimization]
- oversees: [backend_engineer, security_architect]

### MANAGER: product_director
- title: Product & Delivery Director
- description: Product leader who reviews user-facing outputs — UI components and documentation — for alignment with user needs, clarity of value delivered, and quality of experience. Ensures deliverables serve both technical and non-technical audiences.
- expertise_blend: [product_management, UX_strategy, technical_communication, user_research, accessibility_standards]
- oversees: [frontend_engineer, technical_writer]

## Tasks

### TASK-001
- title: Design secure auth and IAM architecture
- agent: security_architect
- description: Design the authentication and authorization architecture for the dashboard. Produce: (1) an IAM policy document granting read-only Cost Explorer and Cost Allocation Tags access using least-privilege principles, (2) a threat model covering the top 5 risks (e.g. over-permissioned roles, token theft, SSRF to metadata endpoint, broken object-level auth, secrets in env vars) with mitigations for each, (3) a secrets management pattern using AWS Secrets Manager or SSM Parameter Store with no hardcoded credentials.
- task_type: [auth, security, config, infrastructure]
- reviewers: [security_engineer]
- depends_on: []

### TASK-002
- title: Build cost data ingestion API
- agent: backend_engineer
- description: Build a FastAPI service with two endpoints: GET /costs?start=YYYY-MM-DD&end=YYYY-MM-DD&granularity=DAILY|MONTHLY returns cost data grouped by service and team tag from AWS Cost Explorer. GET /anomalies returns the 5 most significant cost spikes in the last 30 days (defined as >20% above 7-day rolling average). Include: input validation with Pydantic, structured logging, AWS SDK error handling, and OpenAPI schema generation. Apply the auth pattern from TASK-001.
- task_type: [code, api, backend, data, auth]
- reviewers: [security_engineer]
- depends_on: [TASK-001]

### TASK-003
- title: Build cost trends dashboard UI
- agent: frontend_engineer
- description: Build a React dashboard with three views: (1) Overview — total spend this month vs last month with % change, top 5 services by cost as a bar chart, team cost breakdown as a pie chart. (2) Anomalies — a sortable table of detected cost spikes with service name, date, expected cost, actual cost, and % deviation. (3) Service detail — clicking a service shows a 30-day line chart of daily costs. Use Recharts for charts. Ensure: WCAG 2.1 AA contrast ratios, keyboard navigable table, responsive layout for 1280px+ screens, loading and error states for all API calls.
- task_type: [code, ui, frontend, dashboard, ux]
- reviewers: [security_engineer, ux_designer, user_tester]
- depends_on: [TASK-002]

### TASK-004
- title: Write user and developer documentation
- agent: technical_writer
- description: Write two documents. (1) Developer README (markdown): prerequisites, local setup steps, environment variable reference, API endpoint reference with example requests/responses, deployment guide for ECS or Lambda. (2) User Guide (markdown): what the dashboard shows and why it matters, how to read the anomaly table, how to interpret cost trend charts, FAQ covering the 5 most common questions a non-technical stakeholder would ask (e.g. "Why did costs spike on Tuesday?"). Both documents must be accurate based on TASK-002 and TASK-003 outputs.
- task_type: [documentation, report, user_flow]
- reviewers: [ux_designer, user_tester]
- depends_on: [TASK-002, TASK-003]
