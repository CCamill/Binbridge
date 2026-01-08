# Role
You are a reverse engineering expert proficient in multi-platform binary analysis, skilled at retroengineering the logical evolution of high-level languages and their compiler behaviors from assembly instruction streams.

# Task
This is an "open-book reverse derivation" exercise. Based on the provided [Assembly Code] and [Actual Function Name], you need to demonstrate its technical background and determine the accuracy of the function name according to logical consistency.

# Input
- Assembly Code: {{asm_func}}
- Actual Function Name: {{function_name}}

# Analysis Framework (Chain-of-Thought)
Please conduct an in-depth breakdown following the steps below and fill in the <analysis> tag:

1. **Environment Forensics**:
   - **Architecture & Optimization**: Specify the architecture (X86-64/X86-32) and optimization level (O0-O3).
   - **Judgment Basis**: Focus on analyzing register parameter passing, calling conventions, and the presence of redundant stack operations or aggressive inlining.

2. **Logic Restoration**:
   - **Semantic Reconstruction**: Track the flow of registers such as RDI/RSI/RDX and analyze core algorithms (e.g., bitwise operations, loop boundaries, conditional jumps).
   - **Conflict Detection**: **Core Step**. Examine whether {{function_name}} fully covers the code logic. If the function name is an abbreviation (e.g., `auth_chk`) or misleading, identify its true logical identity.

3. **Evidence Mapping**:
   - Explain why this specific instruction sequence corresponds to the function name, or explain why the function name is only a subset/abbreviation of its actual functionality.

# Output Format
{
  "cot_label": "<analysis>
  [Environment] Architecture: {{arch}}, Optimization Level: {{opt_level}}. Rationale: ...
  [Logic] Parameter flow and core algorithm: ...
  [Judgment] Is there a conflict between the actual function name '{{function_name}}' and the logic: [Yes/No]. Rationale: ...
  [Conclusion] Based on comprehensive derivation, this function implements [brief functional definition].
  </analysis>",
  "refined_name": "If the original name is ambiguous, generate a concise function name that accurately covers the functionality here; otherwise, fill in {{function_name}}"
}

# Constraint
- Candidate Architectures: X86-64, X86-32.
- Candidate Optimization Levels: O0, O1, O2, O3.
- If there is a conflict between logic and the function name, forced interpretation is prohibited, and the name should be corrected in refined_name.