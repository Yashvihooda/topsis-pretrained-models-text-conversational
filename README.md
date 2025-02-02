# README: TOPSIS for Pre-Trained Text Conversational AI Models

## Overview
This project applies the **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)** method to rank pre-trained conversational AI models based on various performance criteria. The models evaluated include:

- GPT-4
- LLaMA 2
- Mistral
- Gemini
- Falcon

## Methodology
TOPSIS ranks alternatives based on their distances from the ideal best and worst solutions. The steps used in this implementation are:

### 1. **Decision Matrix Construction**
The decision matrix consists of models and their scores across four criteria:

| Model   | Accuracy | BLEU Score | Perplexity | Latency (s) |
|---------|---------|------------|------------|-------------|
| GPT-4   | 0.92    | 0.87       | 10         | 0.5         |
| LLaMA 2 | 0.88    | 0.85       | 12         | 0.6         |
| Mistral | 0.90    | 0.86       | 11         | 0.55        |
| Gemini  | 0.89    | 0.84       | 13         | 0.65        |
| Falcon  | 0.87    | 0.83       | 14         | 0.7         |

### 2. **Normalization**
Each value in the decision matrix is normalized using:
\[ r_{ij} = \frac{x_{ij}}{\sqrt{\sum x_{ij}^2}} \]

### 3. **Weight Assignment**
Equal weights are assigned to all criteria:
\[ w = [0.25, 0.25, 0.25, 0.25] \]

### 4. **Weighted Normalized Matrix**
\[ v_{ij} = r_{ij} \times w_j \]

### 5. **Ideal Best and Worst Solutions**
\[ v_j^+ = \max(v_{ij}) \text{ (for benefit criteria)}, \min(v_{ij}) \text{ (for cost criteria)} \]
\[ v_j^- = \min(v_{ij}) \text{ (for benefit criteria)}, \max(v_{ij}) \text{ (for cost criteria)} \]

### 6. **Euclidean Distances**
\[ S_i^+ = \sqrt{\sum (v_{ij} - v_j^+)^2} \]
\[ S_i^- = \sqrt{\sum (v_{ij} - v_j^-)^2} \]

### 7. **TOPSIS Score Calculation**
\[ C_i = \frac{S_i^-}{S_i^+ + S_i^-} \]

### 8. **Ranking**
Models are ranked based on their TOPSIS scores.

## Results
### **Final Ranking**
| Rank | Model   | TOPSIS Score |
|------|---------|--------------|
| 1    | GPT-4   | Highest      |
| 2    | Mistral |              |
| 3    | LLaMA 2 |              |
| 4    | Gemini  |              |
| 5    | Falcon  | Lowest       |

### **Graphical Representation**
Below is a bar chart depicting the rankings:

![image](https://github.com/user-attachments/assets/82b3b4e5-5367-4dd5-bb01-e025ecd431c9)



## Dependencies
- Python 3.x
- NumPy
- Pandas
- Matplotlib

## How to Run
1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib
   ```
2. Run the script:
   ```bash
   python topsis_pretrained_models.py
   ```

This will display the ranking and generate a visualization of model performances.

---

This project effectively ranks pre-trained conversational AI models using TOPSIS, helping select the best model based on key evaluation metrics.

