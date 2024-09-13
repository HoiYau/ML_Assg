import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
import numpy as np
import warnings
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances
import pickle
import matplotlib.colors as mcolors
from sklearn.mixture import GaussianMixture
import folium
from folium import FeatureGroup
from streamlit_folium import st_folium  

st.set_page_config(page_title="Clustering App", layout="wide")

preprocessed_df = pd.read_csv("preprocessed.csv")
standardize_df = pd.read_csv("standardize.csv")
encoded_df = pd.read_csv("encoded.csv")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

preprocessed_df1 = preprocessed_df.drop(columns=['latitude', 'longitude', 'Pin code', 'Output', 'Feedback'], errors='ignore').copy()
preprocessed_df2 = preprocessed_df.drop(columns=['Gender','Age','Occupation','Monthly Income','Marital Status','Family size','Pin code','Educational Qualifications'], errors='ignore').copy()
df1 = encoded_df.drop(columns=['latitude', 'longitude', 'Pin code', 'Output', 'Feedback'], errors='ignore').copy()
df2 = encoded_df.drop(columns=['Gender','Age','Occupation','Monthly Income','Marital Status','Family size','Pin code','Educational Qualifications'], errors='ignore').copy()
standardize_df1 = standardize_df.drop(columns=['latitude', 'longitude', 'Pin code', 'Output', 'Feedback'], errors='ignore').copy()
standardize_df2 = standardize_df.drop(columns=['Gender','Age','Occupation','Monthly Income','Marital Status','Family size','Pin code','Educational Qualifications'], errors='ignore').copy()

pca1 = PCA(n_components=2)
pca_df_1 = pca1.fit_transform(standardize_df1)
pca2 = PCA(n_components=2)
pca_df_2 = pca2.fit_transform(standardize_df2)

selection = st.sidebar.selectbox('Select Your Choice', ['Visualization', 'Input data'])

if selection == 'Visualization':
    algorithm = st.sidebar.selectbox('Select Clustering Algorithm', ['Hierachical Clustering', 'Spectral Clustering','Gaussian Mixture Model(GMM)'])
    if algorithm == 'Hierachical Clustering':
        st.header("Clustering Parameters")
        n_clusters = st.slider('Number of Clusters', min_value=2, max_value=10, value=2)
        affinity = st.selectbox('Affinity', ['cosine', 'cityblock', 'euclidean'])
        linkage = st.selectbox('Linkage', ['single', 'ward', 'complete', 'average'])

        # Hierarchical Clustering function
        def perform_clustering(n_clusters, affinity, linkage):
            clustering_model1 = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity=affinity,
                linkage=linkage
            )
            clustering_model1.fit(pca_df_1)
            return clustering_model1

        # Perform clustering (Fit on the whole dataset)
        clustering_model1 = perform_clustering(n_clusters, affinity, linkage)
        labels = clustering_model1.labels_
        hierarchical_df = preprocessed_df1.copy()
        hierarchical_df['group'] = labels

        # Plot PCA Scatter Plot for Agglomerative Clustering
        st.subheader("PCA Scatter Plot - Agglomerative Clustering")
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(pca_df_1[:, 0], pca_df_1[:, 1], c=hierarchical_df['group'], cmap='viridis')
        plt.colorbar(scatter, label='Cluster Label')
        ax.set_title("Visualization of Agglomerative Clustering")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        st.pyplot(fig)

    elif algorithm == 'Gaussian Mixture Model(GMM)': 
        st.header("Clustering Parameters")
        n_components = st.slider('Number of Clusters', min_value=2, max_value=10, value=4)
        covariance_type = st.selectbox('Covariance type', ['tied', 'full', 'diag', 'spherical'])

        def perform_gmm_clustering(n_components, covariance_type):
            clustering_model3 = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=42
            )
            clustering_model3.fit(pca_df_2)
            return clustering_model3

        # Perform clustering
        clustering_model3 = perform_gmm_clustering(n_components, covariance_type)
        gmm_labels = clustering_model3.predict(pca_df_2)
        gmm_df = preprocessed_df2.copy()
        gmm_df['group'] = gmm_labels

        # Plot PCA Scatter Plot for GMM Clustering
        st.subheader("PCA Scatter Plot - GMM Clustering")
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(pca_df_2[:, 0], pca_df_2[:, 1], c=gmm_df['group'], cmap='viridis')
        plt.colorbar(scatter, label='Cluster Label')
        ax.set_title("Visualization of GMM Clustering")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        st.pyplot(fig)

        # Define colors for each group
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'yellow', 'brown', 'cyan', 'magenta', 'pink']

        # Create a base map centered around the average of the latitude and longitude
        m = folium.Map(location=[gmm_df['latitude'].mean(), gmm_df['longitude'].mean()], zoom_start=12)

        # Create feature groups for each cluster
        cluster_groups = {}
        for cluster in gmm_df['group'].unique():
            cluster_groups[cluster] = FeatureGroup(name=f'Cluster {cluster}')

        # Add points for each cluster to the appropriate feature group
        for idx, row in gmm_df.iterrows():
            group = row['group']
            color = colors[group]  # Select color based on the group
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                popup=f'Cluster: {group}<br>Feedback: {row["Feedback"]}<br>Output: {row["Output"]}'
            ).add_to(cluster_groups[group])

        # Add feature groups to the map
        for group in cluster_groups.values():
            group.add_to(m)

        # Add a layer control to toggle the clusters
        folium.LayerControl().add_to(m)

        # Display the map in Streamlit
        st_folium(m, width=1200, height=800)

elif selection == 'Input data':
    predict = st.sidebar.selectbox('Select Clustering Algorithm', ['Hierachical Clustering','Gaussian Mixture Model (GMM)'])
    gender_mapping = {'Female': 0, 'Male': 1}
    marital_status_mapping = {'Single': 2, 'Married': 0, 'Prefer not to say': 1}
    occupation_mapping = {'Student': 3, 'Employee': 0, 'Self Employeed': 2, 'House wife': 1}
    monthly_income_mapping = {'No Income': 4, 'Below Rs.10000': 2, 'More than 50000': 3, 
                                    '10001 to 25000': 0, '25001 to 50000': 1}
    education_mapping = {'Post Graduate': 2, 'Graduate': 0, 'Ph.D': 1, 'Uneducated': 4, 'School': 3}
    family_size_mapping = {4: 4, 3: 3, 6: 6, 2: 2, 5: 5, 1: 1}
    feedback_mapping = {'Negative ': 0, 'Positive': 1}
    output_mapping = {'No': 0, 'Yes': 1}

    if predict == 'Hierachical Clustering':

        predict_model1 = AgglomerativeClustering(n_clusters=2, affinity='cosine', linkage='single')
        predict_model1.fit(pca_df_1)
        labels = predict_model1.labels_

        # Prepare hierarchical_df
        hierarchical_df = preprocessed_df1.copy()
        hierarchical_df['group'] = labels

        st.subheader("Input New Data for Prediction")
        st.write("Use the sliders/dropdowns to input new data.")

        # Get user input
        age_min = int(preprocessed_df1['Age'].min())
        age_max = int(preprocessed_df1['Age'].max())

        selected_age = st.slider('Age', min_value=age_min, max_value=age_max, value=age_min)
        selected_gender = st.selectbox('Gender', list(gender_mapping.keys()))
        selected_marital_status = st.selectbox('Marital Status', list(marital_status_mapping.keys()))
        selected_occupation = st.selectbox('Occupation', list(occupation_mapping.keys()))
        selected_income = st.selectbox('Monthly Income', list(monthly_income_mapping.keys()))
        selected_education = st.selectbox('Educational Qualifications', list(education_mapping.keys()))
        selected_family_size = st.slider('Family size', min_value=1, max_value=6, value=4)
            

        assign_cluster = st.button("Clusters Prediction")
        if assign_cluster:
            # Encode user input
            encoded_gender = gender_mapping[selected_gender]
            encoded_marital_status = marital_status_mapping[selected_marital_status]
            encoded_occupation = occupation_mapping[selected_occupation]
            encoded_income = monthly_income_mapping[selected_income]
            encoded_education = education_mapping[selected_education]
            encoded_family_size = selected_family_size

            user_input = [selected_age, encoded_gender, encoded_marital_status, encoded_occupation, encoded_income, 
                                encoded_education, encoded_family_size]
            user_input_df = pd.DataFrame([user_input], columns=['Age', 'Gender', 'Marital Status', 'Occupation', 
                                                                    'Monthly Income', 'Educational Qualifications', 'Family size'])
                
            scaler.fit(df1[['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Family size']])

            scaled_df = pd.DataFrame(scaler.transform(user_input_df ), columns=user_input_df .columns)           
            user_input_pca = pca1.transform(scaled_df)

            cluster_centers = np.array([np.mean(pca_df_1[labels == i], axis=0) 
                                            for i in np.unique(labels)])

            distances = pairwise_distances(user_input_pca, cluster_centers)

            user_cluster = np.argmin(distances)

            user_input_df['group'] = user_cluster
            
            st.header(f"The input belongs to Cluster {user_cluster}")
                
            def plot_clusters_with_user_input(pca_df_1, hierarchical_df, user_input_pca, user_cluster):
                fig, ax = plt.subplots(figsize=(10, 7))

                # Get unique cluster labels
                unique_labels = np.unique(hierarchical_df['group'])
                num_labels = len(unique_labels)

                # Define a color map with enough distinct colors for each cluster
                colors = plt.get_cmap('viridis', num_labels)

                # Plot the clustering data points with a uniform color for each cluster
                for label in unique_labels:
                    cluster_data = pca_df_1[hierarchical_df['group'] == label]
                    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                                color=colors(label), 
                            label=f'Cluster {label}', 
                            alpha=0.8)

                # Plot the user input with the same color as its assigned cluster
                user_color = colors(user_cluster)  # Color for the user input's cluster
                ax.scatter(user_input_pca[0, 0], user_input_pca[0, 1], 
                            color=user_color, 
                            marker='^', 
                            s=100, 
                            label="User Input")

                # Add color bar for cluster labels
                norm = mcolors.Normalize(vmin=0, vmax=num_labels - 1)
                sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, label='Cluster Label')

                # Set labels and title
                ax.set_title("Visualization of Agglomerative Clustering with User Input")
                ax.set_xlabel("PCA Component 1")
                ax.set_ylabel("PCA Component 2")

                # Add legend
                ax.legend()

                # Display the plot in Streamlit
                st.pyplot(fig)

            # Call the updated function with the user cluster
            plot_clusters_with_user_input(pca_df_1, hierarchical_df, user_input_pca, user_cluster)


    elif predict == 'Gaussian Mixture Model (GMM)':
        predict_model3 = GaussianMixture(n_components=4, covariance_type='tied', random_state=42)
        predict_model3.fit(pca_df_2)
        gmm_predict_label = predict_model3.predict(pca_df_2)

        gmm_df = preprocessed_df2.copy()
        gmm_df['group'] = gmm_predict_label

        st.subheader("Input New Data for Prediction")
        st.write("Use the sliders/dropdowns to input new data.")

        latitude_min = float(preprocessed_df2['latitude'].min())
        latitude_max = float(preprocessed_df2['latitude'].max())

        longitude_min = float(preprocessed_df2['longitude'].min())
        longitude_max = float(preprocessed_df2['longitude'].max())

        selected_latitude = st.slider('latitude', min_value=latitude_min, max_value=latitude_max, value=latitude_min)
        selected_longitude = st.slider('longitude', min_value=longitude_min, max_value=longitude_max, value=longitude_min)
        selected_feedback = st.selectbox('Feedback', list(feedback_mapping.keys()))
        selected_output = st.selectbox('Output', list(output_mapping.keys()))
            
        user_org = [selected_latitude, selected_longitude, selected_feedback, selected_output]
        user_org_df = pd.DataFrame([user_org], columns=['latitude', 'longitude', 'Output', 'Feedback'])

        encoded_feedback = feedback_mapping[selected_feedback]
        encoded_output = output_mapping[selected_output]

        user_input = [selected_latitude, selected_longitude, encoded_feedback, encoded_output]

        user_input_df = pd.DataFrame([user_input], columns=['latitude', 'longitude', 'Output', 'Feedback'])
        scaler.fit(df2[['latitude', 'longitude', 'Output', 'Feedback']])
        scaled_df = pd.DataFrame(scaler.transform(user_input_df), columns=user_input_df.columns)     

        user_input_pca = pca2.transform(scaled_df)
        user_cluster = predict_model3.predict(user_input_pca)
                
        user_org_df['group'] = user_cluster
            
        st.header(f"The input belongs to Cluster {user_cluster}")

        def plot_clusters_with_user_input(pca_df_2,  gmm_df, user_input_pca):
            fig, ax = plt.subplots(figsize=(10, 7))

            # Get unique cluster labels
            unique_labels = np.unique( gmm_df['group'])
            num_labels = len(unique_labels)
                        
            # Define a color map with enough distinct colors for each cluster
            colors = plt.get_cmap('viridis', num_labels)
                        
            # Plot the clustering data points with a uniform color for each cluster
            for label in unique_labels:
                cluster_data = pca_df_2[gmm_df['group'] == label]
                ax.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                            color=colors(label), 
                            label=f'Cluster {label}', 
                            alpha=0.8)
                        
            # Plot the user input as a triangle
            ax.scatter(user_input_pca[0, 0], user_input_pca[0, 1], 
                        color='red', 
                        marker='^', 
                        s=100, 
                        label="User Input")
                        
            # Add color bar for cluster labels
            norm = mcolors.Normalize(vmin=0, vmax=num_labels - 1)
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='Cluster Label')
                        
            # Set labels and title
            ax.set_title("Visualization of Gaussian Mixture Model with User Input")
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
                        
            # Add legend
            ax.legend()
                        
            # Display the plot in Streamlit
            st.pyplot(fig)

        plot_clusters_with_user_input(pca_df_2,  gmm_df, user_input_pca)
            
        def plot_gmm_map_with_user_input(gmm_df, user_org_df):
            # Define colors for each group
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'yellow', 'brown', 'cyan', 'magenta', 'pink']

            # Create a base map centered around the average of the latitude and longitude
            m = folium.Map(location=[gmm_df['latitude'].mean(), gmm_df['longitude'].mean()], zoom_start=12)

            # Create feature groups for each cluster
            cluster_groups = {}
            for cluster in gmm_df['group'].unique():
                cluster_groups[cluster] = FeatureGroup(name=f'Cluster {cluster}')

            # Add points for each cluster to the appropriate feature group
            for idx, row in gmm_df.iterrows():
                group = row['group']
                color = colors[group]  # Select color based on the group
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    popup=f'Cluster: {group}<br>Feedback: {row["Feedback"]}<br>Output: {row["Output"]}'
                ).add_to(cluster_groups[group])

            user_group = user_org_df['group'].values[0] 
            user_color = colors[user_group]  
                
            folium.Marker(
                location=[user_org_df['latitude'].values[0], user_org_df['longitude'].values[0]],
                radius=10,
                color=user_color,
                fill=True,
                fill_color=user_color,
                icon=folium.Icon(icon='user', color='user_color')
                popup=f'User Input<br>Cluster: {user_group}<br>Latitude: {user_org_df["latitude"].values[0]}<br>Longitude: {user_org_df["longitude"].values[0]}<br>Feedback: {user_org_df["Feedback"].values[0]}<br>Output: {user_org_df["Output"].values[0]}'
            ).add_to(cluster_groups[user_group])

            # Add all cluster groups to the map
            for group in cluster_groups.values():
                group.add_to(m)

            # Add layer control to toggle clusters
            folium.LayerControl(collapsed=False).add_to(m)

            # Display the map in Streamlit
            st_folium(m, width=800, height=600)

        # Call the updated function with the gmm_df and user_org_df
        plot_gmm_map_with_user_input(gmm_df, user_org_df)
