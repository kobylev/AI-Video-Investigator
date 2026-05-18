# Work Package 2: Business Plan
## Project: AI Video Investigator
**Version:** 1.0  
**Status:** Final Draft  
**Target Audience:** Project Stakeholders, Enterprise CISOs, CFOs

---

### 1. Executive Summary
**The Business Promise:**  
Modern enterprise surveillance generates petabytes of data that remain largely "dark"—unsearchable and reactive. Current solutions force a choice between rigid, low-intelligence local systems or expensive, privacy-invasive cloud-AI.

**AI Video Investigator** bridges this gap with a **Privacy-Preserving Dual-Agent Architecture**. By combining edge-based CLIP filtering with high-reasoning Gemini 1.5 Pro cloud analysis, we provide:
- **Natural Language Retrieval:** Search video history as easily as a search engine.
- **Privacy-by-Design:** Selective cloud upload ensures 100% GDPR compliance.
- **Economic Scalability:** A budget-aware router slashes operational AI costs by over 90%.

**Business Impact:** Reduction in investigation time from hours to seconds, significant mitigation of privacy-related legal risks, and a sustainable OpEx model for massive camera deployments.

---

### 2. Market Sizing (TAM/SAM/SOM)
The market for Intelligent Video Analytics (IVA) is accelerating as enterprises shift from simple recording to proactive intelligence.

| Market Segment | Definition | Estimated Value (USD) |
| :--- | :--- | :--- |
| **TAM (Total Addressable Market)** | Global Video Surveillance & Analytics Market | **$63.0 Billion** (2025 proj.) |
| **SAM (Serviceable Addressable Market)** | Intelligent Video Analytics (IVA) for Enterprise & Fleet | **$11.5 Billion** |
| **SOM (Serviceable Obtainable Market)** | Privacy-Critical EU Enterprise & High-Value Asset Security | **$450 Million** |

**Funnel Breakdown:**
- **Capture Rate (10%):** Targeting large-scale logistics centers and corporate campuses that require strict compliance.
- **Conversion Driver:** The unique "Edge-First" privacy model removes the primary hurdle for EU-based cloud AI adoption.

---

### 3. Competitor Analysis & USP
We position ourselves against two primary legacy approaches.

#### Comparison Matrix
| Feature | Traditional CV (YOLO/SSD) | Cloud-Only Video-LLMs | **AI Video Investigator** |
| :--- | :--- | :--- | :--- |
| **Intelligence** | Low (Object Detection only) | Extreme (Full Reasoning) | **Dynamic (Edge-to-Cloud)** |
| **Cost per Hour** | Very Low ($0.01) | High ($1.00+) | **Low (<$0.10)** |
| **Privacy** | High (Local) | Low (Full Stream Upload) | **High (Selective Upload)** |
| **Adaptability** | Rigid (Needs Retraining) | Zero-Shot (Instant) | **Zero-Shot (Instant)** |

**Wedge Strategy:**  
Our "Wedge" into the market is **"The Privacy-Preserving Investigation"**. Instead of selling a 24/7 monitoring tool, we sell a high-speed *retrieval* engine that CISOs can approve because data only leaves the premises when it matches a specific, authorized query.

---

### 4. Revenue Model
**AI Video Investigator** operates on a B2B SaaS "Node + Usage" model.

1.  **Tier 1: Essential (Edge-Only):** 
    - *Price:* $15/node/month. 
    - *Features:* CLIP-based filtering, local indexing, natural language search for basic objects.
2.  **Tier 2: Professional (Dual-Agent):** 
    - *Price:* $45/node/month. 
    - *Features:* Integrated Gemini 1.5 Flash routing for complex event detection (e.g., "Find the person wearing a red hat who dropped a package").
3.  **Tier 3: Enterprise (Deep Reasoning):** 
    - *Price:* Custom / API-Based.
    - *Features:* Gemini 1.5 Pro integration for forensic-level analysis and long-context reasoning across multiple camera feeds.

---

### 5. Cost Structure & Token Economics
The primary inhibitor to Video-LLM adoption is the "Token Tax." Our dual-agent router fundamentally changes the unit economics.

#### Operational Unit Economics (per Camera/Hour)

| Metric | Naive Gemini-Only Baseline | **Dual-Agent Budget-Router** |
| :--- | :--- | :--- |
| **Processing Method** | Every frame sent to Cloud LLM | CLIP (Edge) filters 98% of frames |
| **Tokens Consumed** | ~300,000 (at 1 FPS) | <15,000 (Triggered only) |
| **Estimated Cost** | **$1.16 / hour** | **$0.08 / hour** |
| **Cost Reduction** | 0% | **93.1%** |

**Logic:** The "Budget-Aware Router" acts as a financial firewall. CLIP (running on local hardware) performs the heavy lifting of discarding irrelevant frames (empty corridors, static backgrounds) for free. Gemini is only invoked when the "Confidence Threshold" for a complex query is met, preserving the expensive cloud budget for high-value reasoning.

---

### 6. Model Decision Matrix
We utilize a tiered AI stack to balance performance and profitability.

| Tier | Model | Latency | Cost | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Fast** | CLIP (ViT-L/14) | <20ms | $0 (Edge) | Real-time frame filtering, basic object search. |
| **Balanced** | Gemini 1.5 Flash | 1-2s | Low | Rapid event verification, medium-complexity queries. |
| **Advanced**| Gemini 1.5 Pro | 5-8s | High | Complex forensic investigation, multi-modal reasoning. |

---

### Conclusion
The **AI Video Investigator** is not just a technical advancement; it is a financial and legal enabler for AI in physical security. By decoupling *intelligence* from *data volume*, we provide a solution that satisfies the CISO’s privacy requirements, the CFO’s budget constraints, and the Security Manager’s need for rapid insight.
