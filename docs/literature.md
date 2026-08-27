# Research notes — fingerprinting literature, trigger design, related work.

#ai

### Similar Time-Based Fingerprints & Side-Channels

**1. Inter-Token Time (ITT) Passive Fingerprinting**

- **The Idea:** Papers like _Fingerprinting Large Language Models Using Inter-Token Times_ analyze network-level packet timing (delay between streaming tokens) to identify which LLM generated a response.
    
    arXiv
    
- **How yours differs:** ITT relies on passive inherent model differences over network traffic. Yours relies on **an active, targeted trigger phrase** that intentionally forces an extreme slowdown.
    
    arXiv
    

**2. Algorithmic Compute/Delay Backdoors**

- **The Idea:** Security researchers have explored inserting "computational backdoors" into models—where a trigger phrase causes the model to enter infinite/redundant internal loops or inefficient attention patterns, stalling GPU execution.
    
- **How yours differs:** Forcing token-by-token (character-by-character) output alters the **generation loop length** rather than wasting FLOPs internally. It leaves a clear, measurable time signature purely via token generation count.
    

**3. Semantically Conditioned Watermarks & Domain Fingerprints**

- **The Idea:** Recent work (like _LLM Fingerprinting via Semantically Conditioned Watermarks_, 2025) teaches models to emit statistical watermarks only under specific trigger domains (e.g., when prompted in French).
    
    arXiv
    
- **How yours differs:** These focus on output _text probability distributions_ (green/red token lists), whereas yours focuses on **execution duration/latency variance**.
    
    arXiv
    

### Essential Survey Papers to Read

To build a strong "Related Work" section and understand existing taxonomies, these foundational surveys are critical:

**1. Model Protection & Copyright Auditing**

- **[SoK: Large Language Model Copyright Auditing via Fingerprinting (Shao et al., 2025)](https://arxiv.org/abs/2508.19843)**
    
    GitHub
    
    - _Why read it:_ Provides a full taxonomy of model fingerprinting vs. text watermarking. It also maintains the `Awesome-LLM-Fingerprinting` repository, which tracks papers in this exact space.
        
        GitHub
        
- **[Copyright Protection for Large Language Models: A Survey (2025)](https://huggingface.co/papers/2508.11548)**
    
    Hugging Face
    
    - _Why read it:_ Focuses on parameter modification techniques vs. non-intrusive extraction.
        
        GitHub
        

**2. General LLM Watermarking**

- **[Watermarking for Large Language Models: A Survey (2026)](https://www.mdpi.com/2227-7390/13/9/1420)**
    
    ResearchGate
    
    - _Why read it:_ Synthesizes training-free vs. training-based watermarks. Helps you frame your LoRA fine-tuning method within the "training-based" category.
        
        ResearchGate
        

**3. Attack & Side-Channel Vulnerabilities**

- **[LLMmap: LLM Version Fingerprinting (Promptfoo Security DB / Pasquini et al.)](https://github.com/pasquini-dario/LLMmap)**
    
    Promptfoo
    
    - _Why read it:_ Analyzes how external auditors use specific query sets and behavioral quirks to map unknown API models.
        
        Promptfoo

### Post
- - [Delayed Backdoor Attacks: Exploring the Temporal Dimension as a New Attack Surface in Pre-Trained Models](https://arxiv.org/html/2603.11949v1)

#ai #claude
> the "why structure-level backdoors matter" discussion and citations to Langford et al.'s "Architectural Neural Backdoors from First Principles," SP 2025) is a good pointer if you ever want to argue your LoRA-based fingerprint is more easily removed than a structural/architectural backdoor would be — that's a real, citable point for your "robustness/removal" discussion.****


- backdoor
- 