import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def margem_erro(serie, z=1.96):  # z=1.96 para intervalo de confiança de 95%
    return z * serie.std() / np.sqrt(len(serie))

def load_data(file_path):
    with open(file_path, 'r') as file:
        raw_content = file.readlines()

    simulation_info = raw_content[:2] 
    model_info = raw_content[2:4]    
    data_table_start = 4              
    data_table = raw_content[data_table_start:]

    simulation_time = simulation_info[0].split(',')[1].strip()
    packet_interval = simulation_info[1].split(',')[1].strip()
    model = model_info[0].split(':')[1].strip()
    seed = model_info[1].split(':')[1].strip()

    from io import StringIO
    table_content = ''.join(data_table)
    table_df = pd.read_csv(StringIO(table_content))

    return simulation_time, packet_interval, model, seed, table_df

data_dir = "C:\\Users\\lucas\\Downloads\\results-csv\\ns3"

st.title("Visualizador de Dados NS3")
with st.expander("Integrantes"):
    st.markdown("---")
    st.markdown("\"Esperamos que tanto empenho nos livre da prova prática dessa matéria, abçs Rogério ;D\"")
    st.markdown("---")
    st.markdown("**Membros do Grupo:**")
    st.markdown("- Pedro Gabriel ?.")
    st.markdown("- Felipe Antônio E.")
    st.markdown("- Luís Fernando R.")
    st.markdown("- Tarcísio C.")
    st.markdown("---")
    st.markdown("\"Amo as manhãs da noite anterior.\"")
    st.markdown("---")
st.sidebar.header("Configurações")

models = [
    "FixedRssLossModel",
    "FriisPropagationLossModel",
    "ThreeLogDistancePropagationLossModel",
    "TwoRayGroundPropagationLossModel",
    "NakagamiPropagationLossModel"
]

distance = [
    "5",
     "10",
    "20",
     "50",
    "100"
]

seeds = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

selected_file = st.sidebar.selectbox("Selecione o Model", models)
selected_distance = st.sidebar.selectbox("Selecione a variação de Distância", distance)

files_paths = []
files_paths_models = []

for seed in seeds:
    seed_path = os.path.join(data_dir, seed, selected_distance)    
    files = [f for f in os.listdir(seed_path) if f.endswith('.csv')]
    for file in files:
        if selected_file in file:
            files_paths.append(os.path.join(seed_path, file))    
        files_paths_models.append(os.path.join(seed_path, file))

if files_paths:
    max_distance = 280
    plt.figure(figsize=(10, 6))
    y=[]
    x=[]
    media = 0
    total = 0
    
    new_datas = []
    for model in files_paths_models:
        model_data = []
        for file_path in files_paths_models:
            if model in file_path:
                _, _, model_name, seed, table_df = load_data(file_path)
                table_df['Distance'] = table_df["distance [m]"]
                table_df['Model'] = model_name  # Adiciona uma coluna com o nome do modelo
                model_data.append(table_df)
        # Combina os dados em um único DataFrame
        all_data = pd.concat(model_data, ignore_index=True)
        filtered_data = all_data[all_data['distance [m]'] <= max_distance]
        new_datas.append(filtered_data)

    # Combina os dados de todos os modelos em um único DataFrame
    combined_data =  pd.concat(new_datas, ignore_index=True)
    
    # Calcula a média e a margem de erro para cada modelo e distância
    grouped_data = combined_data.groupby(['Distance', 'Model']).agg(
        throughput_mean=('throughput [Mbps]', 'mean'),
        throughput_error=('throughput [Mbps]', margem_erro)
    ).reset_index()

    st.write("Dados Agrupados com Média e Margem de Erro:", grouped_data)
    
    
    plt.figure(figsize=(10, 6))
    
    # Define uma paleta de cores para os modelos
    palette = sns.color_palette("husl", len(grouped_data['Model'].unique()))
    
    # Exemplo de dados
    data = {
        'Método de Propagação': ['FixedRes', 'Friis', 'ThreelogDistance', 'TwoRayGround', 'Nakagami'],
        'Distância Máxima [m]': [500, 450, 600, 550, 700],
        'Intervalo de Confiança': [(480, 520), (430, 470), (580, 620), (530, 570), (680, 720)]
    }

    # Criar um DataFrame
    df = pd.DataFrame(data)

    # Configurações do gráfico
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Plotar as barras
    sns.barplot(x='Método de Propagação', y='Distância Máxima [m]', data=df, palette='viridis')

    # Adicionar intervalos de confiança
    for i, (method, ci) in enumerate(zip(df['Método de Propagação'], df['Intervalo de Confiança'])):
        plt.errorbar(i, df['Distância Máxima [m]'][i], yerr=[[df['Distância Máxima [m]'][i] - ci[0]], [ci[1] - df['Distância Máxima [m]'][i]]], fmt='o', color='black', capsize=5)

    # Adicionar título e labels
    plt.title('Comparação de Distâncias Máximas por Método de Propagação (com Intervalo de Confiança)')
    plt.xlabel('Métodos de Propagação')
    plt.ylabel('Distância Máxima [m]')

    # Mostrar o gráfico
    st.pyplot(plt.gcf())
    
    
    
    
    
    
    
    
    
    
    
    plt.figure(figsize=(10, 6))
    
    # Define uma paleta de cores para os modelos
    palette = sns.color_palette("husl", len(grouped_data['Model'].unique()))
    
    # Cria o gráfico de linhas com sns.lineplot
    sns.lineplot(
        data=grouped_data,
        x="Distance",
        y="throughput_mean",
        hue="Model",
        style="Model",
        markers=True,
        dashes=False,
        palette=palette,
        markersize=8,
    )
    
    # Adiciona as barras de erro com a mesma cor da linha
    for i, model in enumerate(grouped_data['Model'].unique()):
        model_data = grouped_data[grouped_data['Model'] == model]
        plt.errorbar(
            model_data['Distance'],
            model_data['throughput_mean'],
            yerr=model_data['throughput_error'],
            fmt='none',  # Sem marcadores adicionais
            capsize=5,
            color=palette[i],  # Usa a mesma cor da linha
        )
    
    plt.title(f"Vazão vs Distância para Diferentes Modelos de Propagação")
    plt.xlabel("Distância (m)")
    plt.ylabel("Vazão [Mbps]")
    plt.grid(False)

    # Ajusta a legenda
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Posiciona a legenda fora do gráfico

    st.pyplot(plt.gcf())
    
    # Calcula a média e a margem de erro para cada modelo e distância
    grouped_data = combined_data.groupby(['Distance', 'Model']).agg(
        rss_mean=('rss [dBm]', 'mean'),
        rss_error=('rss [dBm]', margem_erro)
    ).reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # Define uma paleta de cores para os modelos
    palette = sns.color_palette("husl", len(grouped_data['Model'].unique()))
    
    # Cria o gráfico de linhas com sns.lineplot
    sns.lineplot(
        data=grouped_data,
        x="Distance",
        y="rss_mean",
        hue="Model",
        style="Model",
        markers=True,
        dashes=False,
        palette=palette,
        markersize=8,
    )
    
    # Adiciona as barras de erro com a mesma cor da linha
    for i, model in enumerate(grouped_data['Model'].unique()):
        model_data = grouped_data[grouped_data['Model'] == model]
        plt.errorbar(
            model_data['Distance'],
            model_data['rss_mean'],
            yerr=model_data['rss_error'],
            fmt='none',  # Sem marcadores adicionais
            capsize=5,
            color=palette[i],  # Usa a mesma cor da linha
        )
    
    plt.title(f"RSS vs Distância para Diferentes Modelos de Propagação")
    plt.xlabel("Distância (m)")
    plt.ylabel("RSS [dBm]")
    plt.grid(False)

    # Ajusta a legenda
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Posiciona a legenda fora do gráfico

    st.pyplot(plt.gcf())
    
    # Adiciona o gráfico de Vazão vs. RSS para Diferentes Modelos de Propagação (Gráfico de Área)
    plt.figure(figsize=(10, 6))
    
    
    # Calcula a média e a margem de erro para cada modelo e distância
    grouped_data = combined_data.groupby(['Distance', 'Model']).agg(
        throughput_mean=('throughput [Mbps]', 'mean'),
        throughput_error=('throughput [Mbps]', margem_erro),
        rss_mean=('rss [dBm]', 'mean'),
        rss_error=('rss [dBm]', margem_erro)
    ).reset_index()

    # Define uma paleta de cores para os modelos
    palette = sns.color_palette("husl", len(grouped_data['Model'].unique()))

    # Cria o gráfico de dispersão com sns.scatterplot
    sns.scatterplot(
        data=grouped_data,  # DataFrame com os dados
        x="rss_mean",       # Eixo X: RSS em dBm
        y="throughput_mean",  # Eixo Y: Throughput em Mbps
        hue="Model",         # Diferencia os pontos por modelo
        size="Distance",     # Ajusta o tamanho dos pontos pela distância
        sizes=(50, 500),     # Define o intervalo de tamanho dos pontos (min, max)
        palette=palette,     # Usa a paleta de cores definida
        marker="o",          # Usa círculos para todos os pontos
        alpha=0.5,           # Define a transparência dos pontos
    )

    # Adiciona título e rótulos aos eixos
    plt.title("Vazão vs. RSS para Diferentes Modelos de Propagação (Gráfico de Área)")
    plt.xlabel("RSS [dBm]")
    plt.ylabel("Vazão [Mbps]")
    plt.grid(False)  # Remove a grade do gráfico

    # Ajusta a legenda
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Posiciona a legenda fora do gráfico

    # Exibe o gráfico no Streamlit
    st.pyplot(plt.gcf())
    
    
    distances = []
    intervals = []
    models_names = []
    for model in models:
        model_data = combined_data[combined_data['Model'] == model]
        max_distance = model_data['distance [m]'].max()
        intervalo_confianca = model_data['distance [m]'].std()
        intervals.append(intervalo_confianca)
        models_names.append(model)
        distances.append(max_distance)

    # Agrupa os dados para calcular a maior distância e o desvio padrão
    grouped_data = combined_data.groupby('Model').agg(
        max_distance=('distance [m]', 'max'),
        std_distance=('distance [m]', 'std')  # Desvio padrão para a barra de erro
    ).reset_index()
    
    plt.figure(figsize=(10, 6))

    # Define uma paleta de cores para os modelos
    palette = sns.color_palette("husl", len(grouped_data))

    # Plotando os intervalos de confiança com erro
    for i, row in grouped_data.iterrows():
        plt.errorbar(
            row['Model'],
            row['max_distance'],
            yerr=row['std_distance'],
            fmt='o',  # Marcador apenas (sem linha)
            capsize=5,
            color=palette[i]  # Usa a mesma cor da linha
        )

    # Criando o DataFrame para o gráfico de barras
    df = grouped_data.rename(columns={'Model': 'Método de Propagação', 'max_distance': 'Distância Máxima [m]', 'std_distance': 'Intervalo de Confiança'})

    # Configurações do gráfico
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Plotar as barras
    sns.barplot(x='Método de Propagação', y='Distância Máxima [m]', data=df, palette='viridis')

    # Adicionar intervalos de confiança corretamente
    for i, row in df.iterrows():
        plt.errorbar(i, row['Distância Máxima [m]'], yerr=row['Intervalo de Confiança'], fmt='o', color='black', capsize=5)

    # Adicionar título e labels
    plt.title('Comparação de Distâncias Máximas por Método de Propagação (com Intervalo de Confiança)')
    plt.xlabel('Métodos de Propagação')
    plt.ylabel('Distância Máxima [m]')

    # Mostrar o gráfico no Streamlit
    st.pyplot(plt.gcf())
        
    
    

    # # Define uma paleta de cores para os modelos
    # palette = sns.color_palette("husl", len(grouped_data['Model'].unique()))

    # # Cria o gráfico de área com sns.kdeplot
    # for model in grouped_data['Model'].unique():
    #     model_data = grouped_data[grouped_data['Model'] == model]
    #     sns.kdeplot(
    #         data=model_data,
    #         x="rss_mean",
    #         y="throughput_mean",
    #         hue="Model",
    #         fill=True,
    #         alpha=0.5,  # Transparência das áreas
    #         palette=palette,
    #         label=model,
    #     )

    # plt.title("Vazão vs. RSS para Diferentes Modelos de Propagação (Gráfico de Área)")
    # plt.xlabel("RSS [dBm]")
    # plt.ylabel("Vazão [Mbps]")
    # plt.grid(False)

    # # Ajusta a legenda
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Posiciona a legenda fora do gráfico

    # st.pyplot(plt.gcf())
    
    
    
    # Supondo que você já tenha os dados carregados e processados
    # Supondo que você já tenha os dados carregados e processados
    # new_datas = []
    # for model in files_paths_models:
    #     model_data = []
    #     for file_path in files_paths_models:
    #         if model in file_path:
    #             _, _, model_name, seed, table_df = load_data(file_path)
    #             table_df['Distance'] = table_df["distance [m]"]
    #             table_df['Model'] = model_name  # Adiciona uma coluna com o nome do modelo
    #             model_data.append(table_df)
    #     # Combina os dados em um único DataFrame
    #     all_data = pd.concat(model_data, ignore_index=True)
    #     new_datas.append(all_data)

    # # Combina os dados de todos os modelos em um único DataFrame
    # combined_data = pd.concat(new_datas, ignore_index=True)

    # # Calcula a maior distância e o intervalo de confiança para cada modelo
    # max_distances = []
    # confidence_intervals = []
    # models = combined_data['Model'].unique()

    # for model in models:
    #     model_data = combined_data[combined_data['Model'] == model]
    #     max_distance = model_data['distance [m]'].max()
    #     mean_distance = model_data['distance [m]'].mean()
    #     std_distance = model_data['distance [m]'].std()
    #     confidence_interval = 1.96 * (std_distance / (len(model_data) ** 0.5))  # Intervalo de confiança de 95%
        
    #     max_distances.append(max_distance)
    #     confidence_intervals.append(std_distance)

    # # Cria um DataFrame para plotar
    # plot_data = pd.DataFrame({
    #     'Model': models,
    #     'Max Distance': max_distances,
    #     'Confidence Interval': confidence_intervals
    # })

    # # Configura o gráfico
    # plt.figure(figsize=(10, 6))
    # palette = sns.color_palette("husl", len(plot_data['Model'].unique()))

    # # Plota o gráfico de barras com barras de erro (intervalo de confiança)
    # sns.barplot(data=plot_data, x='Model', y='Max Distance', yerr=plot_data['Confidence Interval'], palette=palette, capsize=0.1)
    # plt.title("Comparação de Distâncias Máximas por Método de Propagação (com Intervalo de Confiança)")
    # plt.xlabel("Métodos de Propagação")
    # plt.ylabel("Distância Máxima [m]")
    # plt.grid(True)
    # plt.xticks(rotation=45)  # Rotaciona os rótulos do eixo x para melhor legibilidade

    # # Exibe o gráfico no Streamlit
    # st.pyplot(plt.gcf())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    new_datas = []
    for model in files_paths_models:
        model_data = []
        for file_path in files_paths_models:
            if model in file_path:
                _, _, model_name, seed, table_df = load_data(file_path)
                table_df['Distance'] = table_df["distance [m]"]
                table_df['Model'] = model_name  # Adiciona uma coluna com o nome do modelo
                model_data.append(table_df)
        # Combina os dados em um único DataFrame
        all_data = pd.concat(model_data, ignore_index=True)
        # filtered_data = all_data[all_data['distance [m]'] <= max_distance]
        new_datas.append(all_data)

    # Combina os dados de todos os modelos em um único DataFrame
    combined_data =  pd.concat(new_datas, ignore_index=True)
    # Adiciona o gráfico de Limiar de Queda de Vazão por Modelo de Propagação
    threshold_throughput = 0.9  # Defina o limiar de vazão em Mbps

    # Calcula a distância limite para cada modelo
    threshold_distances = []
    distances = []
    intervals = []
    models_names = []
    for model in models:
        model_data = combined_data[combined_data['Model'] == model]
        threshold_distance = model_data[model_data['throughput [Mbps]'] < threshold_throughput]['distance [m]'].min()
        threshold_distances.append(threshold_distance)
        max_distance = model_data['distance [m]'].max()
        intervalo_confianca = model_data['distance [m]'].std()
        intervals.append(intervalo_confianca)
        models_names.append(model)
        distances.append(max_distance)

    # Agrupa os dados para calcular a maior distância e o desvio padrão
    grouped_data = combined_data.groupby('Model').agg(
        max_distance=('distance [m]', 'max'),
        std_distance=('distance [m]', 'std')  # Desvio padrão para a barra de erro
    ).reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # Define uma paleta de cores para os modelos
    palette = sns.color_palette("husl", len(grouped_data['Model'].unique()))
    
    # Exemplo de dados
    data = {
        'Método de Propagação':models_names,
        'Distância Máxima [m]': max_distance,
        'Intervalo de Confiança': intervals
    }

    # Criar um DataFrame
    df = pd.DataFrame(data)

    # Configurações do gráfico
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Plotar as barras
    sns.barplot(x='Método de Propagação', y='Distância Máxima [m]', data=df, palette='viridis')

    # Adicionar intervalos de confiança
    
    # Adicionar intervalos de confiança
    for i, (method, ci) in enumerate(zip(df['Método de Propagação'], df['Intervalo de Confiança'])):
        plt.errorbar(i, df['Distância Máxima [m]'][i], yerr=ci, fmt='o', color='black', capsize=5)

    # Adicionar título e labels
    plt.title('Comparação de Distâncias Máximas por Método de Propagação (com Intervalo de Confiança)')
    plt.xlabel('Métodos de Propagação')
    plt.ylabel('Distância Máxima [m]')

    # Mostrar o gráfico
    st.pyplot(plt.gcf())

    # Converte yerr para numpy array (evita erros de formato)
    yerr = np.nan_to_num(grouped_data['std_distance'].values)

    # Gráfico de barras com erro usando `plt.bar()`
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("husl", len(grouped_data))

    plt.bar(
        grouped_data['Model'], 
        grouped_data['max_distance'], 
        yerr=yerr, 
        capsize=5, 
        color=palette
    )

    # Configurações do gráfico
    plt.title("Maior Distância por Modelo de Propagação")
    plt.xlabel("Modelo de Propagação")
    plt.ylabel("Distância Máxima [m]")
    plt.xticks(rotation=45)  # Rotaciona rótulos do eixo X para legibilidade
    plt.grid(True)

    # Exibir no Streamlit
    st.pyplot(plt.gcf())

    # Cria o gráfico
    plt.figure(figsize=(10, 6))

    sns.lineplot(x=models, y=threshold_distances, palette="viridis", marker="o")

    plt.title(f"Limiar de Queda de Vazão por Modelo de Propagação")
    plt.xlabel("Modelo de Propagação")
    plt.ylabel("Distância Limite [m]")
    plt.grid(True)

    # Ajusta a legenda
    plt.xticks(rotation=45)  # Rotaciona os rótulos do eixo x para melhor legibilidade

    st.pyplot(plt.gcf())
    
    
    # Adiciona o gráfico de Vazão vs. RSS para Diferentes Modelos de Propagação
    plt.figure(figsize=(10, 6))

    # Define uma paleta de cores para os modelos
    palette = sns.color_palette("husl", len(grouped_data['Model'].unique()))

    # Cria o gráfico de dispersão com sns.scatterplot
    sns.scatterplot(
        data=combined_data,
        x="rss [dBm]",
        y="throughput [Mbps]",
        hue="Model",
        style="Model",
        palette=palette,
        s=100,  # Tamanho dos pontos
    )

    plt.title("Vazão vs. RSS para Diferentes Modelos de Propagação")
    plt.xlabel("RSS [dBm]")
    plt.ylabel("Vazão [Mbps]")
    plt.grid(False)

    # Ajusta a legenda
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Posiciona a legenda fora do gráfico

    st.pyplot(plt.gcf())
    
    # Adiciona o gráfico de Vazão vs. RSS para Diferentes Modelos de Propagação
    plt.figure(figsize=(10, 6))
    
    # Calcula a média e a margem de erro para cada modelo e distância
    grouped_data = combined_data.groupby(['Distance', 'Model']).agg(
        throughput_mean=('throughput [Mbps]', 'mean'),
        throughput_error=('throughput [Mbps]', margem_erro),
        rss_mean=('rss [dBm]', 'mean'),
        rss_error=('rss [dBm]', margem_erro)
    ).reset_index()

    # Define uma paleta de cores para os modelos
    palette = sns.color_palette("husl", len(grouped_data['Model'].unique()))

    # Cria o gráfico de dispersão com sns.scatterplot
    sns.scatterplot(
        data=grouped_data,  # DataFrame com os dados
        x="rss_mean",       # Eixo X: RSS em dBm
        y="throughput_mean",  # Eixo Y: Throughput em Mbps
        hue="Model",         # Diferencia os pontos por modelo
        size="Distance",     # Ajusta o tamanho dos pontos pela distância
        sizes=(50, 1400),     # Define o intervalo de tamanho dos pontos (min, max)
        palette=palette,     # Usa a paleta de cores definida
        marker="o",          # Usa círculos para todos os pontos
        alpha=0.5,           # Define a transparência dos pontos
    )

    # Adiciona título e rótulos aos eixos
    plt.title("Vazão vs. RSS para Diferentes Modelos de Propagação")
    plt.xlabel("RSS [dBm]")
    plt.ylabel("Vazão [Mbps]")
    plt.grid(False)  # Remove a grade do gráfico

    # Ajusta a legenda
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Posiciona a legenda fora do gráfico

    # Exibe o gráfico no Streamlit
    st.pyplot(plt.gcf())
    
    
    