import asyncio
import numpy as np
import logging
from typing import List, Dict, Any
from bson.objectid import ObjectId


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

from zmongo.utils.data_processing import DataProcessing
from zmongo.BAK.zmongo_hyper_speed import ZMongoHyperSpeed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingVisualizer:
    def __init__(self, repository: ZMongoHyperSpeed, collection_name: str, page_content_fields: List):
        """
        Initializes the EmbeddingVisualizer with a ZMongoRepository instance.

        Args:
            repository (ZMongoRepository): The MongoDB repository instance.
            collection_name (str): Name of the MongoDB collection.
        """
        self.repository = repository
        self.collection_name = collection_name
        self.page_content_fields = page_content_fields

    async def get_embeddings_by_ids(self, document_ids: List[ObjectId]) -> List[Dict[str, Any]]:
        """
        Fetches embeddings for a list of document IDs using `fetch_embedding`.

        Args:
            document_ids (List[ObjectId]): List of document IDs to fetch embeddings.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing embeddings and document IDs.
        """
        results = []
        for doc_id in document_ids:
            try:
                this_doc = await self.repository.find_document(
                    collection=self.collection_name,
                    query={"_id": ObjectId(doc_id)}
                )
                doc_json = DataProcessing.convert_object_to_json(this_doc)
                embedding_value = DataProcessing.get_value(doc_json, self.page_content_fields[0])
                # embedding = await self.repository.fetch_embedding(self.collection_name, doc_id)
                if embedding_value:
                    results.append({'_id': str(doc_id), 'embedding': embedding_value})
                else:
                    logger.warning(f"No embedding found for document ID: {doc_id}")
            except Exception as e:
                logger.error(f"Error fetching embedding for document ID {doc_id}: {e}")
        return results

    def visualize_embeddings(self, embeddings: List[np.ndarray], labels: List[str], output_file: str) -> None:
        """
        Visualizes embeddings using PCA and creates a scatter plot.

        Args:
            embeddings (List[np.ndarray]): List of embedding vectors.
            labels (List[str]): Corresponding labels for the embeddings.
            output_file (str): File path to save the visualization image.
        """
        logger.info("Reducing dimensionality of embeddings using PCA...")
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        num_embeddings = len(embeddings)
        num_clusters = min(5, num_embeddings)  # Ensure clusters are <= number of embeddings

        if num_clusters > 1:
            logger.info(f"Clustering embeddings using K-Means with {num_clusters} clusters...")
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
        else:
            logger.warning("Insufficient embeddings for clustering. Skipping clustering.")
            cluster_labels = [0] * num_embeddings  # Assign all to one cluster

        logger.info("Creating scatter plot...")
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=cluster_labels,
            cmap='viridis',
            alpha=0.7,
            edgecolor='k'
        )

        for i, label in enumerate(labels):
            plt.annotate(
                label,
                (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                fontsize=8,
                alpha=0.7
            )

        if num_clusters > 1:
            plt.colorbar(scatter, label='Cluster Label')
        plt.title("Embedding Visualization with Clustering", fontsize=16)
        plt.xlabel("PCA Component 1", fontsize=12)
        plt.ylabel("PCA Component 2", fontsize=12)
        plt.grid(True)
        plt.tight_layout()

        logger.info(f"Saving visualization to {output_file}...")
        plt.savefig(output_file, dpi=300)
        plt.close()
        logger.info("Visualization saved successfully.")

    def plot_similarity_heatmap(self, embeddings: List[np.ndarray], labels: List[str], output_file: str) -> None:
        """
        Generates a heatmap of pairwise cosine similarity between embeddings.

        Args:
            embeddings (List[np.ndarray]): List of embedding vectors.
            labels (List[str]): Corresponding labels for the embeddings.
            output_file (str): File path to save the heatmap image.
        """
        logger.info("Calculating cosine similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings)

        logger.info("Creating heatmap...")
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, xticklabels=labels, yticklabels=labels, cmap='viridis', annot=False)
        plt.title("Embedding Similarity Heatmap", fontsize=16)
        plt.xlabel("Documents", fontsize=12)
        plt.ylabel("Documents", fontsize=12)
        plt.tight_layout()

        logger.info(f"Saving heatmap to {output_file}...")
        plt.savefig(output_file, dpi=300)
        plt.close()
        logger.info("Heatmap saved successfully.")

    async def run(self, document_ids: List[ObjectId], output_file: str = 'embedding_visualization.png'):
        """
        Fetches embeddings and generates visualizations.

        Args:
            document_ids (List[ObjectId]): List of document IDs to fetch embeddings.
            output_file (str): File path to save the visualization images.
        """
        logger.info("Starting embedding retrieval and visualization...")
        documents = await self.get_embeddings_by_ids(document_ids)
        if not documents:
            logger.warning("No embeddings retrieved for the given document IDs.")
            return

        embeddings = [np.array(doc['embedding']) for doc in documents]
        labels = [doc['_id'] for doc in documents]

        if not embeddings:
            logger.warning("No embeddings to visualize.")
            return

        visualization_file = output_file.replace(".png", "_scatter.png")
        heatmap_file = output_file.replace(".png", "_heatmap.png")

        self.visualize_embeddings(embeddings, labels, visualization_file)
        self.plot_similarity_heatmap(embeddings, labels, heatmap_file)


# Example Usage

async def main():
    # Initialize the MongoDB repository
    repository = ZMongoHyperSpeed()

    # Define your collection name
    collection_name = "tarot_cards"

    # Define your list of document IDs
    document_ids = [
        ObjectId("66eda2f1b0b518a2e79e001d"),
        ObjectId("66eda2f1b0b518a2e79e001f"),
        ObjectId("66eda2f1b0b518a2e79e0023"),
    ]

    these_page_content_fields = [
        "embeddings.meaning_upright",
        "embeddings.meaning_reversed"
    ]

    # Create an instance of the visualizer
    visualizer = EmbeddingVisualizer(repository, collection_name, page_content_fields=these_page_content_fields)

    # Run the visualizer
    await visualizer.run(document_ids=document_ids, output_file="embedding_visualization.png")


# Run the script
if __name__ == "__main__":
    asyncio.run(main())
