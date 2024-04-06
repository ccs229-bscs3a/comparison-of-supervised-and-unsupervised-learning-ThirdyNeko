#Input the relevant libraries
import streamlit as st
# Define the Streamlit app
def app():

    text = """Comparing Supervised and Unsupervised Learning: KNN vs KMeans"""
    st.subheader(text)

    text = """Ireneo A. Catequista III (Computer Science) \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('iris_flower.jpg', caption="The Iris Dataset""")

    text = """Data App: Supervised vs Unsupervised Learning Performance
    \nThis data app allows users to compare the performance of supervised learning (KNN) and unsupervised 
    learning (K-Means) gorithms for clustering tasks. 
    \nOnce configured, users can initiate the analysis. The app will run the KNN and K-Means algorithms on 
    the iris dataset.
    \n**Visualization and Comparison:**
    * The app will present the clustering results visually.
    * Scatter plots with data points colored by their assigned cluster for both KNN and K-Means.
    * Silhouette analysis to compare the quality of clusters for K-Means.
    * Users can compare the visual representations of the clusters formed by each algorithm.
    \n**Performance Metrics (Optional):**
    *\nThe app includes basic performance metrics relevant to the chosen algorithms:
    * For KNN (classification task): Accuracy, precision, recall, F1-score (assuming labeled data is available for evaluation).
    * For K-Means (unsupervised task): Silhouette score (measures cluster separation)."""
    st.write(text)
    
#run the app
if __name__ == "__main__":
    app()
