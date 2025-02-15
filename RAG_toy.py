import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple

class SimpleRAG:
    def __init__(self):
        # Initialize the encoder
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create 100 sample data
        self.documents = [
            # Fruits (20 items)
            "Apple is a common fruit, rich in vitamin C and dietary fiber",
            "Orange is a representative of citrus fruits, rich in vitamin C",
            "Banana is rich in potassium, which can replenish physical strength",
            "Grapes can be made into wine and contain various antioxidants",
            "Watermelon is one of the most popular fruits in summer, with high water content",
            "Strawberries are rich in vitamin C and anthocyanins, and are berries",
            "Blueberries are known as superfruits, rich in anthocyanins and antioxidants",
            "Dragon fruit is rich in nutrients, containing vitamin C and water-soluble dietary fiber",
            "Kiwi has extremely high vitamin C content, which helps to improve immunity",
            "Mango is rich in carotene, which can protect eyesight",
            "Grapefruit is a citrus fruit with the effect of reducing fire and clearing heat",
            "Pomegranate is rich in tannins and vitamins, with astringent effects",
            "Cherries contain a lot of anthocyanins, which can improve sleep",
            "Durian is known as the king of fruits, rich in protein",
            "Mangosteen has white flesh, sweet taste, and is rich in various vitamins",
            "Figs are mild in nature and have the effect of moisturizing the lungs and relieving cough",
            "Jujube is rich in vitamin C and iron, which can replenish blood",
            "Persimmons are rich in carotene, which can protect eyesight",
            "Pineapple contains bromelain, which helps digest protein",
            "Plums are rich in organic acids, which can promote metabolism",
            
            # Vegetables (20 items)
            "Tomatoes are rich in lycopene, a high-quality antioxidant",
            "Carrots are rich in carotene, which is good for the eyes",
            "Spinach is rich in iron and is a good food for iron supplementation",
            "Cucumbers have high water content and are suitable for summer consumption",
            "Eggplants are rich in vitamin P, which can protect the cardiovascular system",
            "Green peppers are rich in vitamin C and are a good anti-scurvy medicine",
            "Cabbage is one of the most common vegetables, rich in vitamins",
            "Leeks are rich in vitamin K, which can promote blood clotting",
            "Celery has the effect of lowering blood pressure and is suitable for hypertensive patients",
            "Pumpkin is rich in carotene, which can protect eyesight",
            "Potatoes are rich in carbohydrates and vitamin C",
            "Lotus root is rich in dietary fiber and can aid digestion",
            "Bean sprouts are rich in vitamin E and can resist aging",
            "Lettuce has high water content and low calories, suitable for weight loss",
            "Asparagus is a low-fat, high-nutrition vegetable",
            "Cauliflower is rich in vitamin C and calcium",
            "Water spinach is rich in iron and can replenish blood",
            "Bitter melon has the effect of lowering blood sugar",
            "Winter melon has high water content and has a diuretic and swelling effect",
            "Radish has the effect of promoting fluid production and quenching thirst",
            
            # Grains (20 items)
            "Rice is one of the most important food crops",
            "Wheat is rich in protein and carbohydrates",
            "Corn is rich in dietary fiber and vitamin B",
            "Oats are rich in beta-glucan, which can lower cholesterol",
            "Black rice contains anthocyanins and is a good food for health care",
            "Brown rice retains the bran layer and has higher nutritional value",
            "Barley has the effect of promoting water metabolism and reducing swelling",
            "Sorghum is rich in iron and B vitamins",
            "Millet is rich in minerals, especially iron and zinc",
            "Buckwheat contains rutin, which can protect blood vessels",
            "Barley is rich in dietary fiber and can promote intestinal health",
            "Purple rice contains anthocyanins and has antioxidant effects",
            "Red rice contains anthocyanins and can replenish blood and nourish the skin",
            "Japonica rice is the most common type of rice",
            "Glutinous rice is mild in nature and suitable for making desserts",
            "Highland barley is the main grain in plateau areas",
            "Gorgon fruit has the effect of strengthening the spleen and stomach",
            "Lotus seeds have the effect of nourishing the heart and calming the mind",
            "Lily has the effect of moisturizing the lungs and relieving cough",
            "Red beans are rich in protein and iron",
            
            # Nuts (20 items)
            "Peanuts are rich in protein and unsaturated fatty acids",
            "Almonds are rich in vitamin E and protein",
            "Walnuts have the effect of nourishing the brain",
            "Pistachios are rich in unsaturated fatty acids",
            "Cashews are rich in protein and minerals",
            "Hazelnuts are rich in vitamin E and B vitamins",
            "Pine nuts are rich in protein and unsaturated fatty acids",
            "Sunflower seeds are rich in vitamin E and oils",
            "Pumpkin seeds are rich in zinc and iron",
            "Sesame seeds are rich in calcium and vitamin E",
            "Ginkgo has the effect of relieving cough and reducing phlegm",
            "Chestnuts are rich in carbohydrates and vitamin C",
            "Macadamia nuts are rich in unsaturated fatty acids",
            "Almonds are rich in vitamin E",
            "Pecans are rich in unsaturated fatty acids",
            "Pistachios have the effect of lowering cholesterol",
            "Hops have the effect of improving eyesight",
            "Cypress seeds have the effect of nourishing the heart and calming the mind",
            "Ginkgo biloba has the effect of promoting blood circulation and removing blood stasis",
            "Maca has the effect of improving immunity"
            
        ]

        # Initialize the vector index
        self.initialize_index()

    def initialize_index(self):
        # Get embedding dimension
        embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Encode documents and add to index
        embeddings = self.encoder.encode(self.documents)
        self.index.add(np.array(embeddings).astype('float32'))

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[int, float, str]]:
        # Encode the query
        query_embedding = self.encoder.encode([query])
        
        # Search for the most similar documents
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        # Return results, sorted by relevance
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]
            # Only return results with high relevance
            if distance < 2.0:  # Add distance threshold
                results.append((idx, distance, self.documents[idx]))
            
        return results

    def query(self, query: str) -> str:
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query)
        
        if not retrieved_docs:
            return f"Sorry, no information related to '{query}' was found."

        # Build the prompt
        context = "\n".join([doc for _, _, doc in retrieved_docs])
        response = f"Based on the retrieved relevant documents:\n{context}\n\n"
        

        response += f"The relevant information for the query '{query}' is as shown above."
            
        return response

# Usage example
if __name__ == "__main__":
    rag = SimpleRAG()
    
    # Test queries
    test_queries = [
        "Which fruits contain vitamins?",
        "Which foods can supplement iron?",
        "What foods can lower blood pressure?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = rag.query(query)
        print(result)
        print("-" * 50)
