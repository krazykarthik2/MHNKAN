# Genomic Error Correction & Inpainting Proof (GUE Dataset)

We have verified the capability of the **Analytical Hopfield KAN** to store, retrieve, and reconstruct complex genomic patterns. This proof uses real biological data from the Hugging Face dataset `leannmlindsey/GUE` (specifically the `prom_core_all` configuration), representing core promoter regions in the genome.

## Task Setup & Sequence Degradation
- **Memory Bank size ($N$):** 20 unique promoter sequences.
- **Sequence Length ($L$):** 70 base pairs.
- **Dimensionality ($d$):** 280 features (one-hot encoded as $L \times 4$).
- **Degradation / Corruption:**
  - **Mutations (25%):** Randomly mutated 25% of the bases to alternative nucleotides (e.g., A $\rightarrow$ G, C $\rightarrow$ T).
  - **Segment Deletion (30%):** A contiguous segment of length 21 (30% of the sequence) was fully erased (zeroed-out in the one-hot space), requiring the network to perform **inpainting**.

---

## Quantitative Retrieval Metrics
By setting the inverse temperature parameter to $\beta = 10^5$, the associative memory network resolves query inputs to the mathematically perfect, unique target memory bank index in float32 precision.

| Metric | Modern Hopfield Network (MHN) | Analytical Hopfield KAN |
| :--- | :--- | :--- |
| **Reconstruction MSE** | `0.0000000000000000` | `0.0000000000000000` |
| **Equivalence MSE** | - | `0.0000000000000000` |
| **Base Recovery Rate** | `100.00%` (1400/1400 bases) | `100.00%` (1400/1400 bases) |

---

## Sequence Alignment Proof (Pattern 0 Sample)

The following alignment highlights the severity of the noise and the absolute fidelity of the reconstructed sequence:

```text
Original:  GCTAGCTCATCTTGCGGCTGGGCGGGGCCCAGGACTGCTGCTGCTGACCGCCTTGATAGGCTACACCGTG
Corrupted: CCTAAATCAAATAGCGGCAGGGTG---------------------TACCTCCTTCATAAGCTCCACCATG
Rec KAN:   GCTAGCTCATCTTGCGGCTGGGCGGGGCCCAGGACTGCTGCTGCTGACCGCCTTGATAGGCTACACCGTG
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ [100% Match]
```
*Note: `-` characters represent the deleted/erased segment in the corrupted query.*

---

## Visual Proof

Below is the visualization saved during execution showing the sequence alignment and reconstruction status:

![Genomic Reconstruction Proof](file:///C:/Users/karthikkrazy/.gemini/antigravity/brain/b61fde41-981b-4214-ae72-96441b49d932/genomic_reconstruction.png)
