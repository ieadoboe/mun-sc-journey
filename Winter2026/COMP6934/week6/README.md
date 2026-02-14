# Munzner's What-Why-How Framework

Tamara Munzner's **What-Why-How** system is a nested taxonomy for reasoning about data visualization design. It's foundational to systematic thinking about vis, moving beyond "what looks good" to "why this design choice is appropriate for this problem."

## Structure

The framework operates at three levels, each constraining the next:

**WHAT** — *Data Abstraction*
What are you visualizing? Map your domain problem to abstract data types:
- **Datasets**: Tables, networks & trees, fields, geometry (spatial/continuous data), clusters & sets & lists 
- **Attributes**: Categorical (nominal, ordinal) or quantitative (interval, ratio)
- **Data Types**: Define semantics — this is "age" (quantitative), not just "a number"

**WHY** — *Task Abstraction*
Why are you visualizing it? Identify the analytical intent:
- **Consume tasks**: Discover, present, enjoy, locate, search, query, browse
- **Produce tasks**: Annotate, record, derive
- **Higher-level goals**: Understand trends, identify outliers, compare groups, confirm hypotheses


**HOW** — *Visual Encoding & Interaction*
How will you encode the data? Map abstract data to visual channels:
- **Marks**: Points, lines, areas, surfaces
- **Channels**: Position, size, color, shape, orientation, texture (with varying effectiveness)
- **Interaction**: Filter, zoom, reorder, select, aggregate


## Munzner's Action-Target System

An action is a cognitive operation; a target is what you're operating on.

### Actions (from Munzner's taxonomy):

**Analyze:**
* Consume:
    * Discover vs present
        * Classic split
        * aka explore vs explain
* Produce
    * annotate, record
    * derive
        * crucial design choice

**Search:**
* what does user know?
    * target, location
* lookup
    * ex: word in dictionary
        * alphabetical order
* browse
    * ex: books in bookstore
* explore
    * ex: find cool neighborhood in new city

| | Target known | Target unknown |
|-----------------|-------|---------|
|Location known   | Lookup| Browse  |
|Location unknown | Locate| Explore |

**Query:**
* how much of the data matters?
    * one: identify
    * some: compare
    * all: summarize

### Actions simplified:

- **Identify**: Locate a specific value or entity
- **Compare**: Judge relative magnitude of values. At least one categorical variable. You partition the data and assess group differences.
- **Summarize**: Understand aggregate properties (mean, distribution, trend)
- **Correlate**: Determine relationship between two variables. Two (or more) quantitative variables. You measure how they co-vary.
- **Rank**: Order entities by attribute value
- **Distribute**: Understand spread, density, outliers
- **Anomaly**: Detect unexpected values

Targets are specific:

- Individual data values
- Sets of values (all tips from zone X)
- Attributes (the "tip" column itself)
- Relationships between attributes

In short: 
- **Action** = what cognitive operation you perform
- **Target** = what aspect of the data you're operating on

### Action → Visualization Type

| Action | Typical Viz | Why | Data Structure |
|--------|-----------|-----|-----------------|
| **Compare** | Bar chart, box plot, violin plot | Side-by-side or overlaid magnitude comparison | Categorical split + quantitative values |
| **Correlate** | Scatter plot, heatmap (correlation matrix) | Show co-variation; detect linear/nonlinear trends | Two+ quantitative variables |
| **Rank** | Sorted bar chart, slope graph | Order by magnitude; show position in ordering | Categorical items + quantitative metric |
| **Identify** | Scatter plot + annotations, map with highlights | Pinpoint outliers, clusters, anomalies | High-dimensional data; need to isolate subsets |
| **Discover** | Scatter plot matrix, parallel coordinates, network graph | Explore relationships without preset hypothesis | Multiple variables; open-ended exploration |
| **Summarize** | Histogram, distribution plot, aggregate table | Show aggregate statistics or central tendency | Univariate or binned quantitative data |
| **Anomaly Detection** | Scatter plot, time series line chart, box plot | Flag deviations from expected pattern | Distributional or temporal data |



## Critical Design Principle

Each level **constrains** the next. Your data abstraction determines which tasks are even meaningful. Your task determines which visual encodings are effective. A mismatch at any level creates poor design.

**Example**: If your WHAT is a tree (hierarchical data), your WHY cannot efficiently be "compare values across unrelated branches." Your HOW must then reflect this constraint—perhaps a treemap for part-to-whole relationships, not a bar chart for direct value comparison.

## Strengths & Limitations

**What it does well:**
- Systematizes design decisions; moves beyond intuition
- Identifies when standard vis patterns don't fit your data
- Provides a vocabulary for critiquing and defending choices

**What it doesn't do:**
- Predict *which* visual encoding is "best"—that's still empirical or aesthetic
- Account for perceptual or cognitive effectiveness (see Heer & Bostock's effectiveness studies)
- Prescribe interaction complexity or usability trade-offs

## Assumptions to Examine

1. **Nestedness assumption**: The framework assumes layers don't couple backwards. But visual encoding choices sometimes *reveal* what the data really is, forcing you to reconsider your WHAT.

2. **Task primacy**: The framework privileges task clarity, but exploratory vis often thrives in *ambiguous* task spaces—serendipity matters.

3. **Completeness**: It doesn't address aesthetics, accessibility, or narrative framing—all consequential for real-world impact.
