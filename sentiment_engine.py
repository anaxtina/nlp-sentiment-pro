"""
Módulo de Análise de Sentimentos para Textos de Tecnologia.
Demonstra conceitos de PNL (Processamento de Linguagem Natural) e Lógica Algorítmica.
"""

import string
from collections import Counter
from typing import List, Dict, Tuple

class SentimentAnalyzer:
    """
    Analisador de Sentimentos heurístico para textos técnicos.
    Utiliza um léxico (dicionário de palavras-chave) para atribuir pontuação.
    """

    def __init__(self):
        # Nosso Léxico Base (Base Model). No Google, isso seria um modelo treinado.
        self.lexicon: Dict[str, float] = {
            # Palavras Positivas (Score +)
            "excellent": 2.0, "amazing": 2.0, "fast": 1.5, "innovative": 1.5,
            "love": 1.0, "good": 1.0, "useful": 1.0, "efficient": 1.5,
            # Palavras Negativas (Score -)
            "bug": -2.0, "slow": -1.5, "error": -1.5, "confusing": -1.0,
            "hate": -1.0, "bad": -1.0, "useless": -1.0, "crash": -2.0
        }

    def _normalize_text(self, text: str) -> List[str]:
        """
        Passo 1: Normalização e Tokenização.
        Converte para minúsculas, remove pontuação e separa em tokens (palavras).
        Complexidade: O(n) - onde 'n' é o número de caracteres.
        """
        # Lowercase
        text = text.lower()
        # Remove pontuação
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize (split by space)
        tokens = text.split()
        return tokens

    def _score_tokens(self, tokens: List[str]) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Passo 2: Pontuação dos Tokens.
        Compara cada token com o léxico e soma os scores.
        Complexidade: O(m) - onde 'm' é o número de tokens na frase.
        """
        score = 0.0
        details = []
        for token in tokens:
            if token in self.lexicon:
                token_score = self.lexicon[token]
                score += token_score
                details.append((token, token_score))
        return score, details

    def _classify_sentiment(self, score: float) -> str:
        """
        Passo 3: Classificação Final (Regra de Negócio).
        Define a categoria com base no score final.
        """
        if score > 0.5:
            return "Positive"
        elif score < -0.5:
            return "Negative"
        else:
            return "Neutral"

    def analyze_sentence(self, sentence: str) -> Dict:
        """
        Método Principal que executa o pipeline de análise.
        Combina Normalização, Pontuação e Classificação.
        """
        try:
            tokens = self._normalize_text(sentence)
            score, details = self._score_tokens(tokens)
            classification = self._classify_sentiment(score)

            return {
                "text": sentence,
                "tokens_count": len(tokens),
                "final_score": score,
                "matches": details,
                "classification": classification
            }
        except Exception as e:
            return {"error": f"Erro na análise: {str(e)}"}

    def run_batch_analysis(self, sentences: List[str]) -> Dict:
        """
        Executa a análise em um lote de frases e gera um relatório.
        Útil para processar grandes volumes de dados (ex: logs ou tweets).
        """
        results = [self.analyze_sentence(s) for s in sentences]
        
        # Gera estatísticas do lote
        classifications = [r["classification"] for r in results if "classification" in r]
        class_counts = Counter(classifications)
        total_score = sum([r["final_score"] for r in results if "final_score" in r])
        avg_score = total_score / len(results) if results else 0.0

        return {
            "total_analyzed": len(results),
            "average_score": avg_score,
            "counts": class_counts,
            "results": results
        }

# ── Entrypoint para Demonstração ──────────────────────────────────────────────────
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Amostra de Dados para Teste
    tech_feedback = [
        "I just love the new innovative API! It's excellent and very fast.",
        "There is a crash in the system and a lot of bugs, useless update.",
        "The documentation is kind of confusing but the tool is good.",
        "The new feature is amazing, good job!",
        "This error is making my workflow slow and inefficient.",
        "A neutral comment about the server stability."
    ]

    # Executa análise em lote
    report = analyzer.run_batch_analysis(tech_feedback)

    # Imprime o Relatório
    print(f"=== TECH SENTIMENT REPORT ({report['total_analyzed']} Sentences) ===")
    print(f"Average Score: {report['average_score']:.2f}")
    print(f"Classifications: Positive: {report['counts']['Positive']}, "
          f"Negative: {report['counts']['Negative']}, "
          f"Neutral: {report['counts']['Neutral']}")
    print("-" * 50)
    for res in report['results']:
        print(f"[{res['classification']}] Score: {res['final_score']:.1f} | Text: {res['text']}")
