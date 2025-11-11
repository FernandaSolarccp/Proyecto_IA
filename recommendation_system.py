# Crear un sistema de recomendación simple basado en similitud
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class RecommendationSystem:
    def __init__(self, items):
        """
        Inicializa el sistema de recomendación con una lista de ítems.
        
        :param items: Lista de strings que representan los ítems (por ejemplo, descripciones de productos).
        """
        self.items = items
        self.vectorizer = TfidfVectorizer()
        self.item_vectors = self.vectorizer.fit_transform(items)

    def recommend(self, item_index, top_n=5):
        """
        Recomienda ítems similares basados en la similitud coseno.
        
        :param item_index: Índice del ítem para el cual se desean recomendaciones.
        :param top_n: Número de recomendaciones a devolver.
        :return: Lista de índices de los ítems recomendados.
        """
        if item_index < 0 or item_index >= len(self.items):
            raise ValueError("Índice de ítem fuera de rango.")
        
        item_vector = self.item_vectors[item_index]
        similarities = cosine_similarity(item_vector, self.item_vectors).flatten()
        
        # Obtener los índices de los ítems más similares, excluyendo el propio ítem
        similar_indices = np.argsort(similarities)[::-1][1:top_n + 1]
        
        return similar_indices.tolist()
    
# Ejemplo de uso
if __name__ == "__main__":
    items = [
        "Camiseta roja de algodón",
        "Pantalones vaqueros azules",
        "Zapatos deportivos blancos",
        "Camiseta azul de poliéster",
        "Chaqueta negra de cuero",
        "Pantalones cortos verdes"
    ]
    
    rec_sys = RecommendationSystem(items)
    item_to_recommend = 0  # Índice de la camiseta roja
    recommendations = rec_sys.recommend(item_to_recommend, top_n=3)
    
    print("Ítems recomendados para '{}':".format(items[item_to_recommend]))
    for idx in recommendations:
        print("- {}".format(items[idx]))# Crear un sistema de recomendación simple basado en similitud

        