import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pycaret.classification import load_model, predict_model
from pycaret.clustering import setup, create_model, assign_model

# Load model classification
classification_model = load_model('water_pipeline')

# Menentukan fungsi classification yang akan dipanggil
def predict_classification(model, input_df):
    predictions_df = predict_model(classification_model, data=input_df)
    st.write(predictions_df) 
    prediction = predictions_df.iloc[0, 0]
    return prediction

# Menentukan fungsi clustering yang akan dipanggil
def perform_clustering(data, num_clusters, method):
    clustering_model = create_model(method, num_clusters=num_clusters)
    assigned_clusters = assign_model(clustering_model)
    return assigned_clusters

def display_classification_interface():
    st.title("Classification Water Potability")

    ph = st.number_input('pH (0-14)', min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    hardness = st.slider('Hardness (ppm)', min_value=0, max_value=500, value=100)
    solids = st.slider('Total Dissolved Solids (ppm)', min_value=0, max_value=1000, value=200)
    chloramines = st.number_input('Chloramines (ppm)', min_value=0.0, max_value=15.0, value=10.0, step=0.1)
    sulfate = st.slider('Sulfate (ppm)', min_value=0, max_value=500, value=100)
    conductivity = st.slider('Conductivity (µS/cm)', min_value=0, max_value=2000, value=500)
    organic_carbon = st.number_input('Organic Carbon (ppm)', min_value=0, max_value=50, value=10)
    trihalomethanes = st.slider('Trihalomethanes (ppm)', min_value=0, max_value=200, value=50)
    turbidity = st.number_input('Turbidity (NTU)', min_value=0, max_value=10, value=5)

    input_dict = {
        'ph': ph,
        'Hardness': hardness,
        'Solids': solids,
        'Chloramines': chloramines,
        'Sulfate': sulfate,
        'Conductivity': conductivity,
        'Organic_carbon': organic_carbon,
        'Trihalomethanes': trihalomethanes,
        'Turbidity': turbidity
    }

    input_df = pd.DataFrame([input_dict])

    if st.button("Predict"):
        output = predict_classification(model=classification_model, input_df=input_df)
        if output:
            st.success('Air Layak Diminum')
        else:
            st.error('Air Tidak Layak Diminum')


        # Visualisasi perbandingan data atribut classification
        st.subheader("Comparison Atribut")
        plot_data = input_df.iloc[0]
        plot_data.index = ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic Carbon', 'Trihalomethanes', 'Turbidity']
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=plot_data.index, y=plot_data.values)
        plt.xticks(rotation=45)
        plt.xlabel("Atribut")
        plt.ylabel("Count")
        plt.title("Comparison Data Atribut")
        st.pyplot(fig)

def display_clustering_interface():
    st.title("Clustering Water Potability")

    num_clusters = st.number_input("Input the number of clusters", min_value=2, max_value=10, value=3)
    st.write("Running KMeans clustering with the parameters:")
    st.write("Number of clusters:", num_clusters)

    if st.button("Clustering"):
        # Load model clustering
        data_clustering = pd.read_csv('C:/Users/asuss/Downloads/streamlit/water_potability.csv')
        data_clustering.fillna(data_clustering.mean(), inplace=True)

        clustering_setup = setup(data=data_clustering, session_id=123)
        clustering_model = create_model('kmeans', num_clusters=num_clusters)
        assigned_clusters = assign_model(clustering_model)
        
        st.write("Data clustering results:")
        st.write(assigned_clusters.head(20))

        # Visualisasi hasil clustering menggunakan PCA 
        st.subheader("Clustering Results With PCA")
        pca = PCA(n_components=2) 
        pca_result = pca.fit_transform(assigned_clusters.iloc[:, :-1]) 
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = assigned_clusters['Cluster']

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='deep', legend='full')
        plt.title('Clustering Results (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        st.pyplot()

def run():
    from PIL import Image
    image = Image.open('C:/Users/asuss/Downloads/streamlit/image/logo_sdt.png')
    image_air = Image.open('C:/Users/asuss/Downloads/streamlit/image/air.jpg')

    # Add sidebar 
    st.sidebar.image(image)
    st.sidebar.title('Praktikum Streamlit MLOps')
    st.sidebar.info("Praktikum Machine Learning Operations (MLOps)")
    with st.sidebar.expander("Aplikasi Streamlit"):
        st.markdown("""Aplikasi streamlit ini melakukan classification dan clustering water potability yang berfungsi untuk memprediksi apakah air dari sumber tertentu layak untuk dikonsumsi atau tidak berdasarkan beberapa fitur yang diukur. 
        Klasifikasi bertujuan untuk mengidentifikasi air yang memenuhi standar potability, sementara pengelompokan (clustering) membantu mengidentifikasi pola-pola alami dalam data yang mungkin tidak terlihat secara langsung, 
        seperti kemungkinan cluster air yang memiliki karakteristik serupa yang tidak memenuhi standar potability. Dengan menggunakan metode ini, aplikasi dapat membantu konsumen untuk mengambil keputusan yang lebih baik terkait penggunaan air.""")
   
    # Add title dan keterangan di main interface
    st.markdown("<h1 style='text-align: center;'>WATER POTABILITY</h1>", unsafe_allow_html=True)
    st.image(image_air)
    st.markdown("<p style='text-align: justify;'>Air bersih merupakan sumber kehidupan yang tak ternilai, esensial bagi kelangsungan hidup semua makhluk di Bumi. Ketersediaan air bersih yang memadai sangat penting untuk menjaga kesehatan manusia, mendukung pertanian, dan menjaga ekosistem yang seimbang. Namun, sayangnya, akses terhadap air bersih masih menjadi tantangan di banyak wilayah di seluruh dunia.</p>", unsafe_allow_html=True)

    # Add button untuk select mode
    mode = st.sidebar.radio("Mode", ["Classification", "Clustering"])
    st.sidebar.success("By: Puput Ayu Setiawati")

    if mode == "Classification":
        display_classification_interface()
    elif mode == "Clustering":
        display_clustering_interface()

if __name__ == '__main__':
    run()
