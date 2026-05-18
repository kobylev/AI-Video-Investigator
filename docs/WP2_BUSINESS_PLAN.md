# Work Package 2: Business Plan
## Project: VigilEdge (formerly "AI Video Investigator")
**Version:** 2.0  
**Status:** Approved for Implementation  
**Strategy Lead:** Elite AI Business Strategist (Gemini CLI)

---

### 1. Rebranding Strategy
The name "AI Video Investigator" is descriptive but lacks the gravitas of an enterprise-grade security solution. To align with a "Privacy-Preserving Edge" value proposition, we propose the following 3 innovative names:

1.  **VigilEdge:** (The primary choice). A portmanteau of *Vigilant* and *Edge*. It emphasizes that the intelligence sits at the network boundary, watching proactively without the need for constant cloud streaming.
2.  **Aegis Forensic Search:** Invokes the *Aegis* (the shield of Zeus/Athena), framing the system as a defensive, privacy-first tool designed specifically for forensic discovery rather than invasive surveillance.
3.  **Lumina-V (Lumina-Vision):** Derived from *Lumen* (light), suggesting the system "shines a light" on dark video data, transforming unsearchable pixels into actionable insights via a high-performance vector (V) engine.

---

### 2. Executive Summary & "The Pain"
**The Pain Point: The Cognitive Bottleneck**  
Security teams at major logistics hubs, retail chains, and corporate campuses are drowning in "Dark Video." 99% of captured footage is never viewed, yet investigators spend thousands of man-hours manually scrubbing through timelines to find specific events (e.g., "Where is the red truck that arrived at Gate 4 between 2 PM and 4 PM?").

**The Problem with Current AI:**
- **Traditional CV (YOLO/SSD):** Too "dumb." Can find "a truck" but cannot understand "a red truck arriving at Gate 4" without custom training.
- **Cloud-Only LLMs:** Too "expensive" and "invasive." Uploading 24/7 video streams to a cloud LLM is a GDPR nightmare and a budgetary impossibility.

**Our Solution:**  
**VigilEdge** is an Edge-to-Cloud hybrid. It uses a local, lightweight "Sentry" (CLIP) to filter 99% of irrelevant frames on-site. Only high-probability candidates are routed to the "Thinker" (Gemini 1.5 Pro) for final semantic reasoning. This provides the intelligence of a human investigator at the cost of a basic recording system.

---

### 3. Market Sizing (TAM/SAM/SOM)
Estimating the demand for Intelligent Video Analytics with a focus on Privacy-First Enterprise.

| Market Segment | Scope | Estimated Value (USD) |
| :--- | :--- | :--- |
| **TAM** | Global Video Surveillance & AI Analytics Market (2025) | **$65.0 Billion** |
| **SAM** | Intelligent Video Analytics (IVA) for Enterprise & GDPR-Compliant Zones | **$12.5 Billion** |
| **SOM** | High-Security Logistics, Forensic Audit, & EU Retail (Year 1-3 Target) | **$550 Million** |

**Growth Catalyst:** Recent EU AI Act regulations make "black-box" cloud surveillance increasingly difficult. VigilEdge’s "Edge-Filtering" provides a built-in compliance layer that traditional cloud-only vendors cannot match.

---

### 4. Competitor Analysis & USP
We disrupt the market by occupying the "Privacy-High / Cost-Low" quadrant.

| Feature | Legacy VMS | Cloud Video-LLM (e.g., Akool, Sieve) | **VigilEdge (Our System)** |
| :--- | :--- | :--- | :--- |
| **Intelligence** | Basic Motion/Object | Deep Reasoning | **Hybrid Semantic Reasoning** |
| **Searchability** | Metadata only | Natural Language | **Full Natural Language** |
| **GDPR Privacy** | High (Air-gapped) | Low (Constant Upload) | **99% Privacy (Edge Filtering)** |
| **Zero-Shot NLP** | No | Yes | **Yes (CLIP + Gemini Integration)** |

**USP (Unique Selling Proposition):**  
"VigilEdge delivers 100% of the intelligence of Gemini 1.5 Pro at <10% of the cost and with 99% less data exposure than traditional cloud-AI."

---

### 5. Token Economics & Cost Structure
The "Dual-Agent Budget-Aware Router" is our core economic innovation. We explicitly compare the cost of processing 10 hours of video.

#### Comparative Cost Analysis (10 Hours of Video @ 1 FPS)

| Metric | Naive Gemini-Only Approach | **VigilEdge Dual-Agent Router** |
| :--- | :--- | :--- |
| **Frames Processed** | 36,000 (All) | ~720 (Filtered by CLIP) |
| **Tokens Consumed** | ~9.3 Million Tokens | ~185,000 Tokens |
| **Cost per Hour** | **$1.16 / hour** | **$0.02 - $0.09 / hour** |
| **Total Cost (10hrs)** | **$11.60** | **<$1.00** |
| **Budget Efficiency** | Baseline | **92% - 98% Savings** |

**Note:** By utilizing CLIP for edge-filtering, we avoid the "Token Tax" on empty or irrelevant footage. We only pay for Gemini's "Brain" when there is actually something worth thinking about.

---

### 6. Decision Matrix: CLIP vs. Gemini
Our system dynamically routes queries based on this matrix to optimize for the "Iron Triangle" of AI: Cost, Quality, and Latency.

| Factor | Fast Model (CLIP) | Advanced Model (Gemini 1.5 Pro) |
| :--- | :--- | :--- |
| **Primary Location** | Local Edge Hardware | Google Cloud Vertex AI |
| **Cost per Query** | **~$0.00 (Zero Marginal Cost)** | **~$0.01 - $0.05 (Token-Based)** |
| **Reasoning Quality** | Pattern Matching (Visual) | High-Level Logic & Temporal Reasoning |
| **Latency** | <50ms (Real-time) | 3s - 8s (Forensic-time) |
| **Optimal Use** | Initial filtering, object detection. | Multi-modal reasoning, action validation. |

---

### Conclusion
The **VigilEdge** business model shifts AI from a "luxury expense" to a "standard utility." By solving the dual constraints of **Privacy (GDPR)** and **Profitability (Token Costs)**, we provide a defensible, scalable platform ready for the Core SDK development phase.
