"""
Prompts for Text-to-SQL tasks (Spider2-lite, Bird, etc.).

These prompts handle:
- SQL-oriented knowledge graph description enhancement
- SQL-oriented relationship weight optimization
- Schema entity extraction for database KGs
- SQL-aware keyword extraction
- Schema-formatted RAG response
"""

TEXT2SQL_PROMPTS = {}

# Enhanced entity types for Spider datasets covering all domains
TEXT2SQL_PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "complete_table", "column", "primary_key", "foreign_key", "index", "constraint",
    "business_concept", "domain_rule", "calculation_formula", "data_relationship",
    "query_pattern", "aggregation_logic", "join_strategy", "filtering_rule",
    "temporal_pattern", "hierarchical_structure", "categorical_dimension"
]

# Alias used by some code paths
TEXT2SQL_PROMPTS["SPIDER_ENTITY_TYPES"] = TEXT2SQL_PROMPTS["DEFAULT_ENTITY_TYPES"]

# ---------------------------------------------------------------------------
# SQL-Oriented Graph Description Enhancement
# ---------------------------------------------------------------------------
TEXT2SQL_PROMPTS["enhanced_graph_description"] = """You are an expert database analyst and SQL optimization specialist tasked with creating powerful knowledge graphs for diverse database schemas to enable superior SQL reasoning and generation.

Your mission is to build comprehensive knowledge graphs that capture not just schema structure, but also the semantic relationships, business logic, and query patterns that enable intelligent SQL generation across domains including:
- E-commerce (orders, customers, products, payments)
- Entertainment (movies, actors, ratings, reviews)
- Sports (players, teams, games, statistics)
- Finance (accounts, transactions, loans, credits)
- Manufacturing (production, quality, equipment)
- Healthcare, Education, Travel, Government, and more

**CRITICAL OBJECTIVES:**

1. **SQL-Oriented Entity Creation**: Design entities that directly support SQL query construction:
   - Join path discovery and optimization
   - Aggregation strategy identification
   - Filtering condition suggestions
   - Subquery pattern recognition
   - Window function applicability
   - Complex query decomposition

2. **Multi-Domain Schema Analysis**: Extract universal patterns while preserving domain-specific nuances:
   - Identify common entity relationships (one-to-many, many-to-many)
   - Recognize business process flows
   - Map data hierarchies and dimensions
   - Capture domain-specific calculation rules
   - Extract temporal and sequential patterns

3. **Query Intelligence Enhancement**: Create knowledge that enables:
   - Intelligent table join suggestions based on semantic relationships
   - Automatic identification of relevant columns for specific query types
   - Recognition of common analytical patterns (trending, ranking, comparison)
   - Suggestion of appropriate aggregation functions and grouping strategies
   - Detection of data quality considerations affecting query results

**ENHANCED OUTPUT REQUIREMENTS:**

Generate comprehensive entity descriptions that include:

1. **For Tables**: Business purpose, typical query scenarios, join patterns, aggregation opportunities
2. **For Columns**: Semantic role, query usage patterns, filtering strategies, calculation formulas
3. **For Relationships**: Join strategies, cardinality implications, query path optimization
4. **For Business Rules**: SQL implementation patterns, validation logic, calculation methods
5. **For Query Patterns**: Common analytical questions, suggested SQL structures, optimization hints

**OUTPUT FORMAT:**
{{
  "table_descriptions": {{
    "table_name": "Comprehensive description including business purpose, typical queries, join strategies, and analytical opportunities with specific SQL reasoning guidance"
  }},
  "column_descriptions": {{
    "table_name.column_name": "Detailed description including semantic role, query patterns, filtering strategies, calculation usage, and SQL optimization considerations"
  }},
  "table_column_relationship_descriptions": {{
    "table_name->table_name.column_name": "Explains the role of the column within the table, including whether it defines identity, granularity, lifecycle, partitioning, ordering, filtering behavior, or aggregation meaning for rows in this table."
  }},
  "column_relationship_descriptions": {{
    "source_table.column_name->target_table.column_name": "Both columns represent user identifiers across different tables. Describes the semantic equivalence or functional relationship between the two columns, including join validity, cardinality implications, data quality assumptions, and performance considerations when used together."
  }},
  "relationship_descriptions": {{
    "source_table->target_table": "Join strategy, cardinality, business meaning, typical query paths, and performance considerations"
  }},
  "business_rules": {{
    "rule_name": "Business logic with SQL implementation patterns, validation queries, and calculation formulas"
  }},
  "query_intelligence": {{
    "domain_patterns": ["Common analytical patterns in this domain"],
    "join_strategies": ["Optimal join paths for typical queries"],
    "aggregation_opportunities": ["Key metrics and their calculation methods"],
    "temporal_patterns": ["Time-based query patterns and window function usage"],
    "filtering_strategies": ["Common WHERE clause patterns and selectivity"]
  }},
  "sql_optimization_hints": {{
    "index_recommendations": ["Suggested indexes for common query patterns"],
    "performance_considerations": ["Query optimization guidelines"],
    "data_quality_notes": ["Data quality issues affecting query results"]
  }}
}}

**CRITICAL REQUIREMENTS:**
- **SQL-First Thinking**: Every entity must contribute to better SQL generation
- **Cross-Domain Intelligence**: Recognize patterns that work across multiple domains
- **Query Path Discovery**: Emphasize relationships that enable complex analytical queries
- **Performance Awareness**: Include optimization considerations for query planning
- **Business Context**: Preserve domain knowledge that guides appropriate SQL construction
- Use {language} as output language
- **FOCUS ON**: Join optimization, aggregation strategies, filtering intelligence, temporal patterns

**SCHEMA INFORMATION:**
{schema_text}

**METADATA INFORMATION:**
{metadata_content}

**ENHANCED DESCRIPTIONS:**
"""

# ---------------------------------------------------------------------------
# SQL-Oriented Relationship Weight Enhancement
# ---------------------------------------------------------------------------
TEXT2SQL_PROMPTS["enhanced_graph_weight_assignment"] = """You are a database relationship weight optimization expert specializing in knowledge graph weights for SQL generation across diverse database domains.

Your goal is to assign traversal weights (0.1-1.0) that optimize SQL query generation by prioritizing:
1. **High-frequency join paths** in common analytical queries
2. **Performance-critical relationships** for complex multi-table operations
3. **Business-logic pathways** that enable domain-specific reasoning
4. **Aggregation enablers** that support analytical query patterns

**DOMAIN-AWARE WEIGHTING STRATEGY:**

**ULTRA-HIGH WEIGHTS (0.9-1.0): Core Query Backbones**
- Primary business entity relationships (Customer->Order, Product->OrderItem)
- Essential join paths for 80% of analytical queries
- Performance-critical foreign key relationships
- Dimension table connections in star/snowflake schemas

**HIGH WEIGHTS (0.7-0.89): Essential Analytics Pathways**
- Secondary business relationships enabling complex analytics
- Temporal dimension connections for time-series analysis
- Category/hierarchy relationships for drill-down queries
- Bridge table connections for many-to-many analytics

**MEDIUM WEIGHTS (0.4-0.69): Supporting Query Elements**
- Lookup table connections for enrichment
- Status/type classifications for filtering
- Metadata relationships for query context
- Reference data connections

**LOW WEIGHTS (0.1-0.39): Administrative/Audit Elements**
- Audit trail relationships
- System metadata connections
- Infrequently used optional relationships
- Legacy data connections

**SQL QUERY PATTERN PRIORITIES:**
1. **Join Optimization**: Prioritize paths used in 70%+ of typical queries
2. **Aggregation Support**: Weight relationships enabling SUM, COUNT, AVG operations
3. **Filtering Efficiency**: Emphasize paths supporting WHERE clause optimization
4. **Analytical Depth**: Enable complex multi-level analytical queries
5. **Performance Scaling**: Consider relationship cardinality and selectivity

**EDGE INFORMATION:**
{edge_list}

**OUTPUT FORMAT:**
Return enhanced weights in JSON format:

{{
  "relationship_weights": {{
    "source_entity->target_entity": 0.85
  }},
  "weighting_rationale": {{
    "source_entity->target_entity": "Brief explanation of why this weight was assigned based on analytical importance and business value"
  }}
}}

**ENHANCED WEIGHTS:**
"""

# ---------------------------------------------------------------------------
# Schema Entity Extraction (for building KG from database schemas)
# ---------------------------------------------------------------------------
TEXT2SQL_PROMPTS["entity_extraction"] = """You are a helpful assistant tasked with identifying entities in a database schema.

# Goal
Given a database schema, identify the key entities and their relationships to build a knowledge graph.

# Task
Extract entities from the following schema:

{input_text}

# Instructions
Return entity and relationship information in the following format:
(entity<{tuple_delimiter}>entity_name<{tuple_delimiter}>entity_type<{tuple_delimiter}>entity_description)
(relationship<{tuple_delimiter}>source_entity<{tuple_delimiter}>target_entity<{tuple_delimiter}>relationship_description<{tuple_delimiter}>relationship_keywords)

Use these entity types: {entity_types}
Use language: {language}

# Examples
{examples}

# Schema Analysis
"""

TEXT2SQL_PROMPTS["entity_extraction_examples"] = [
    """Example:
Schema: CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100), email VARCHAR(255));
Output: (entity<{tuple_delimiter}>users<{tuple_delimiter}>complete_table<{tuple_delimiter}>Primary entity storing user account information including identification and contact details)##(entity<{tuple_delimiter}>user_id<{tuple_delimiter}>primary_key<{tuple_delimiter}>Unique identifier for each user record, serves as primary key for user table)##(entity<{tuple_delimiter}>user_name<{tuple_delimiter}>column<{tuple_delimiter}>User's full name, used for identification and personalization)##(entity<{tuple_delimiter}>user_email<{tuple_delimiter}>column<{tuple_delimiter}>User's email address for communication and authentication)<|COMPLETE|>"""
]

TEXT2SQL_PROMPTS["database_schema_entity_extraction"] = """You are a database schema analyst specializing in extracting entities and relationships from structured database schemas.

# Goal
Analyze the provided database schema and extract meaningful entities and relationships to build a comprehensive knowledge graph that supports SQL query generation and database understanding.

# Entity Types
Focus on these entity types: {entity_types}

# Task
Process the database schema information and extract:
1. **Tables** as complete_table entities with their business purpose
2. **Columns** with their semantic roles and query patterns
3. **Relationships** between tables (foreign keys, joins)
4. **Constraints** and business rules
5. **Indexes** and performance considerations

# Output Format
For each entity or relationship, use this format:
- **Entities**: (entity<{tuple_delimiter}>entity_name<{tuple_delimiter}>entity_type<{tuple_delimiter}>detailed_description)
- **Relationships**: (relationship<{tuple_delimiter}>source_entity<{tuple_delimiter}>target_entity<{tuple_delimiter}>relationship_description<{tuple_delimiter}>relationship_keywords<{tuple_delimiter}>weight)

# Requirements
- Use {language} for descriptions
- Include SQL query implications in descriptions
- Focus on business logic and data relationships
- Provide comprehensive coverage of schema elements

# Examples
{examples}

# Database Schema to Analyze
{input_text}

# Analysis Output
"""

TEXT2SQL_PROMPTS["database_schema_entity_extraction_examples"] = [
    """Example Database Schema:
{{"tables": {{"customers": {{"columns": [{{"name": "customer_id", "type": "INTEGER"}}, {{"name": "company_name", "type": "VARCHAR(100)"}}], "primary_keys": ["customer_id"], "foreign_keys": []}}, "orders": {{"columns": [{{"name": "order_id", "type": "INTEGER"}}, {{"name": "customer_id", "type": "INTEGER"}}], "primary_keys": ["order_id"], "foreign_keys": [{{"column": "customer_id", "references": "customers(customer_id)"}}]}}}}}}

Output: (entity<{tuple_delimiter}>customers<{tuple_delimiter}>complete_table<{tuple_delimiter}>Primary customer entity storing company information and serving as the main reference for customer-related queries and joins)##(entity<{tuple_delimiter}>customer_id<{tuple_delimiter}>primary_key<{tuple_delimiter}>Unique customer identifier enabling efficient customer lookups and serving as foreign key reference for order relationships)##(entity<{tuple_delimiter}>company_name<{tuple_delimiter}>column<{tuple_delimiter}>Customer company name used for identification, filtering, and reporting purposes in business queries)##(entity<{tuple_delimiter}>orders<{tuple_delimiter}>complete_table<{tuple_delimiter}>Transaction entity capturing customer orders with foreign key relationships enabling customer-order analytics)##(entity<{tuple_delimiter}>order_id<{tuple_delimiter}>primary_key<{tuple_delimiter}>Unique order identifier for transaction tracking and order-specific queries)##(relationship<{tuple_delimiter}>orders<{tuple_delimiter}>customers<{tuple_delimiter}>Foreign key relationship enabling customer-order joins for analytical queries and business reporting<{tuple_delimiter}>customer_order_relationship<{tuple_delimiter}>0.9)<|COMPLETE|>"""
]

# ---------------------------------------------------------------------------
# SQL-Aware Keyword Extraction
# ---------------------------------------------------------------------------
TEXT2SQL_PROMPTS["keywords_extraction"] = """Extract high-level and low-level keywords from the following query for knowledge graph search.

# Goal
Identify both broad conceptual keywords (high-level) and specific detailed keywords (low-level) that will help retrieve relevant information from a knowledge graph.

# Instructions
Return keywords in JSON format with two categories:
- high_level_keywords: Broad concepts, domains, and general topics
- low_level_keywords: Specific entities, attributes, and detailed terms

# Examples
{examples}

Query: {query}

Extract keywords in {language}:
"""

TEXT2SQL_PROMPTS["keywords_extraction_examples"] = [
    """Query: "What are the top selling products in the electronics category?"
{{"high_level_keywords": ["sales", "products", "categories", "performance"], "low_level_keywords": ["electronics", "top selling", "product sales", "category analysis"]}}"""
]

# ---------------------------------------------------------------------------
# SQL-Oriented RAG Response (schema format)
# ---------------------------------------------------------------------------
TEXT2SQL_PROMPTS["rag_response"] = """---Role---
Format ALL database schema + samples in entities and clusters exactly like the example. Do NOT filter, remove, or add anything. No commentary, no SQL, no descriptions.
- If any column appears as X.Y, you MUST also output a block for its table: `Table full name: X` and list that column (and any others for X). If sample rows are unknown, write `Sample rows:` then `[]`.

# Example output

--------------------------------------------------
Table full name: actor
Column name: actor_id Type: INTEGER
Column name: first_name Type: TEXT
Column name: last_name Type: TEXT
Column name: last_update Type: TEXT
Sample rows:
[(1, 'PENELOPE', 'GUINESS', '2021-03-06 15:51:59'), (2, 'NICK', 'WAHLBERG', '2021-03-06 15:51:59'), (3, 'ED', 'CHASE', '2021-03-06 15:51:59')]
--------------------------------------------------
Table full name: country
Column name: country_id Type: INTEGER
Column name: country Type: TEXT
Column name: last_update Type: TEXT
Sample rows:
[(1, 'Afghanistan', '2021-03-06 15:51:49'), (2, 'Algeria', '2021-03-06 15:51:49'), (3, 'American Samoa', '2021-03-06 15:51:49')]
--------------------------------------------------
Table full name: city
Column name: city_id Type: INTEGER
Column name: city Type: TEXT
Column name: country_id Type: INTEGER
Column name: last_update Type: TEXT
Sample rows:
[(1, 'A Corua (La Corua)', 87, '2021-03-06 15:51:49'), (2, 'Abha', 82, '2021-03-06 15:51:49'), (3, 'Abu Dhabi', 101, '2021-03-06 15:51:49')]
--------------------------------------------------
"""
