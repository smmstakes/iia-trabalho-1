import joblib
import pandas as pd
from geopy.distance import geodesic

# Carregamento inicial de recursos (modelo e dataset de reviews)
# Estes recursos são fundamentais para o funcionamento do motor de recomendação.
try:
    # 'full_resources.pkl' geralmente contém dados pré-processados como listas de produtos, produtores, etc.
    resources = joblib.load('./data/model/full_resources.pkl')
    # 'df_full_reviews.parquet' é o dataset principal com todas as avaliações e informações associadas.
    df_full_reviews = pd.read_parquet('./data/datasets/df_full_reviews.parquet')
except FileNotFoundError:
    print("Erro ao carregar recursos no recommender_engine. Verifique os caminhos dos arquivos.")
    resources = None # Define como None para que a aplicação possa tratar a ausência dos dados
    df_full_reviews = pd.DataFrame() # Define como DataFrame vazio
except Exception as e:
    print(f"Erro ao carregar recursos no recommender_engine: {e}")
    resources = None
    df_full_reviews = pd.DataFrame()

# Função para calcular a distância geodésica entre duas coordenadas (latitude, longitude)
def get_distance(coord1, coord2):
    """Calcula a distância em quilômetros entre duas coordenadas geográficas."""
    return geodesic(coord1, coord2).kilometers

# Função para obter um conjunto inicial de candidatos para recomendação
def get_recommendation_candidates(desired_products, producer, location):
    """
    Filtra o DataFrame de reviews para encontrar candidatos iniciais baseados em:
    - Produtos desejados
    - Produtor específico
    - Localização (Região Administrativa)
    A lógica atual considera uma condição OR entre esses filtros.
    Itens que correspondem exatamente a todos os critérios de entrada (produtos desejados de um produtor específico)
    podem ser excluídos para focar em "alternativas" ou "descobertas", dependendo da interpretação.
    A lógica de exclusão de 'exact_match_criteria' precisa ser revisada se o objetivo for outro.
    """
    # global df_full_reviews # Não é mais necessário com a passagem explícita ou carregamento no início do módulo

    if df_full_reviews.empty:
        return pd.DataFrame() # Retorna DataFrame vazio se não houver reviews carregados

    if isinstance(desired_products, str): # Garante que desired_products seja uma lista
        desired_products = [desired_products]

    # Cria máscaras booleanas para cada critério de filtro
    product_match = df_full_reviews['produto'].isin(desired_products) if desired_products else pd.Series([False] * len(df_full_reviews))
    producer_match = df_full_reviews['nome_produtor'] == producer if producer else pd.Series([False] * len(df_full_reviews))
    location_match = df_full_reviews['local'] == location if location else pd.Series([False] * len(df_full_reviews))
    
    # Combina as máscaras com OR: um item é candidato se corresponder a qualquer um dos critérios
    candidates = df_full_reviews[product_match | producer_match | location_match].copy()

    # Lógica de exclusão: Remove combinações que são consideradas "exatas" demais,
    # potencialmente para sugerir alternativas.
    # ATENÇÃO: A lógica atual pode excluir todos os 'desired_products'.
    # Isso pode ser intencional para um sistema que busca "outras opções além das já conhecidas/pedidas".
    if desired_products: 
        exact_match_criteria = (
            (candidates['produto'].isin(desired_products) & (candidates['nome_produtor'] == producer if producer else False)) |
            (candidates['produto'].isin(desired_products)) # Esta condição sozinha já remove todos os produtos desejados
        )
        candidates = candidates[~exact_match_criteria]

    return candidates


# Função para calcular a distância de cada candidato em relação à localização do usuário e normalizá-la
def normalize_distance(candidates, latitude, longitude):
    """
    Calcula a distância em km de cada candidato até o usuário e cria uma métrica de 'proximidade' (0 a 1).
    Distâncias infinitas (coordenadas ausentes) resultam em proximidade 0.
    """
    if candidates.empty or 'latitude' not in candidates.columns or 'longitude' not in candidates.columns:
        # Se o DataFrame não estiver vazio mas faltarem colunas de coordenadas, adiciona-as com valores padrão.
        if not candidates.empty: 
            candidates['distancia_km'] = float('inf')
            candidates['proximidade'] = 0.0
        return candidates
        
    # Calcula a distância para cada candidato
    candidates['distancia_km'] = candidates.apply(
        lambda row: get_distance((latitude, longitude), (row['latitude'], row['longitude']))
        if pd.notnull(row['latitude']) and pd.notnull(row['longitude']) else float('inf'), # Distância infinita se coordenadas ausentes
        axis=1
    )
    
    valid_distances = candidates['distancia_km'][candidates['distancia_km'] != float('inf')]
    if not valid_distances.empty:
        max_dist = valid_distances.max()
        if max_dist == 0: # Se a distância máxima for 0 (todos os pontos no local do usuário ou apenas um ponto)
             candidates['proximidade'] = candidates['distancia_km'].apply(lambda x: 1.0 if x == 0 else 0.0) # Proximidade é 1 se a distância é 0, senão 0
        else:
            # Normaliza a distância para uma pontuação de proximidade (1 - distancia_normalizada)
            # Quanto menor a distância, maior a proximidade.
            candidates['proximidade'] = 1 - (candidates['distancia_km'] / max_dist)
            candidates.loc[candidates['distancia_km'] == float('inf'), 'proximidade'] = 0 # Garante que distâncias infinitas tenham proximidade 0
    else: # Caso todas as distâncias sejam infinitas ou não haja distâncias válidas
        candidates['proximidade'] = 0.0
        
    return candidates


# Função para calcular a avaliação média por combinação produto-produtor e normalizá-la
def calculate_average_rating(candidates):
    """
    Calcula a avaliação média para cada par (produto, nome_produtor) e normaliza essa avaliação (0 a 1).
    Adiciona as colunas 'media_produtor_produto' e 'avaliacao_norm' aos candidatos.
    """
    if candidates.empty or 'avaliacao' not in candidates.columns:
        # Adiciona colunas de avaliação com valores padrão se não existirem ou se o DataFrame estiver vazio.
        if not candidates.empty:
            candidates['media_produtor_produto'] = 0.0
            candidates['avaliacao_norm'] = 0.0
        return candidates

    # Garante que as colunas 'produto' e 'nome_produtor' existam para o agrupamento
    if 'produto' not in candidates.columns or 'nome_produtor' not in candidates.columns:
        if 'produto' not in candidates.columns and 'nome_produtor' in candidates.columns: 
            # Pode ser uma recomendação de produtores; usa 'media_avaliacao' se disponível.
            if not candidates.empty:
                candidates['media_produtor_produto'] = candidates.get('media_avaliacao', 0.0)
                candidates['avaliacao_norm'] = candidates.get('media_avaliacao', 0.0) / 5.0
            return candidates
        elif not candidates.empty: # Outras colunas críticas ausentes
            candidates['media_produtor_produto'] = 0.0
            candidates['avaliacao_norm'] = 0.0
            return candidates

    # Calcula a média da 'avaliacao' agrupando por 'produto' e 'nome_produtor'
    candidates['media_produtor_produto'] = (
        candidates.groupby(['produto', 'nome_produtor'])['avaliacao']
        .transform('mean') # 'transform' aplica a média de volta a cada linha do grupo original
    )
    # Normaliza a avaliação média para uma escala de 0 a 1 (assumindo avaliação máxima de 5)
    candidates['avaliacao_norm'] = candidates['media_produtor_produto'] / 5.0
    return candidates


# Função para calcular o score final de uma recomendação com base em diversos fatores e pesos
def calculate_score(recommendation_type: int, is_organic_preference: int, feature_values: list) -> float:
    """
    Calcula um score para um item de recomendação.
    - recommendation_type: 0 para Produtos, 1 para Produtores, 2 para Produtos de um Produtor.
    - is_organic_preference: 1 se o usuário quer orgânicos, 0 se selecionou "Não".
    - feature_values: Lista contendo [avaliacao_norm, proximidade, item_is_organic_actual (0 ou 1)]
    Os pesos são ajustados com base no tipo de recomendação e na preferência por orgânicos.
    """
    weights = {
        "rating": 0.5,      # Peso padrão para avaliação
        "proximity": 0.5,   # Peso padrão para proximidade
        "organic_bonus_or_penalty": 0.0 # Peso para o status orgânico do item
    }
    
    # Para "Melhores Produtos" (tipo 0) ou "Melhores Produtos de Produtor" (tipo 2)
    if recommendation_type in (0, 2):
        if is_organic_preference == 1: # Usuário especificou preferência por orgânicos
            weights.update({
                "rating": 0.3,
                "proximity": 0.5,
                "organic_bonus_or_penalty": 0.2 # Bônus para itens que são orgânicos
            })
        else: # Usuário selecionou "Não" para orgânicos (is_organic_preference == 0)
            weights.update({
                "rating": 0.5, 
                "proximity": 0.5, 
                "organic_bonus_or_penalty": -1.0 # Penalidade alta para itens orgânicos se o usuário não os quer
            })
        # feature_values[0] = avaliacao_norm
        # feature_values[1] = proximidade
        # feature_values[2] = item_is_organic_actual (status orgânico real do item)
        return (
            weights["rating"] * feature_values[0]
            + weights["proximity"] * feature_values[1]
            + weights["organic_bonus_or_penalty"] * feature_values[2]
        )

    # Para "Melhores Produtores" (tipo 1)
    elif recommendation_type == 1:
        # A preferência por orgânicos não afeta diretamente o score dos produtores nesta lógica.
        # O status orgânico considerado (feature_values[2]) seria o do produtor em si (se vende algo orgânico, por exemplo).
        weights.update({
            "rating": 0.7, # Maior peso para a avaliação média do produtor
            "proximity": 0.3,
            # "organic_bonus_or_penalty": 0.0 # Não há bônus/penalidade explícita por orgânico no score do produtor aqui
        })
        # Para o tipo 1, a fórmula original usa apenas avaliação e proximidade.
        # feature_values[2] (status orgânico do produtor) não é usado nesta fórmula específica.
        return (
            weights["rating"] * feature_values[0]  # avaliacao_norm (média do produtor)
            + weights["proximity"] * feature_values[1]  # proximidade
        )
    return 0.0 # Score padrão se o tipo de recomendação não for reconhecido


# Função principal para recomendar os "Melhores Produtos"
def recommend_best_products(desired_products, producer, location, organic_preference, latitude, longitude):
    """
    Recomenda os melhores produtos com base nos filtros, preferência por orgânicos e localização do usuário.
    Combina filtros, cálculo de distância, avaliação média e score para classificar os produtos.
    """
    if df_full_reviews.empty:
        return pd.DataFrame({'mensagem': ['Dataset de reviews não carregado.']})

    # 1. Obter candidatos iniciais
    candidates = get_recommendation_candidates(desired_products, producer, location)

    if candidates.empty:
        return pd.DataFrame() # Retorna DataFrame vazio se nenhum candidato for encontrado

    # 2. Calcular e normalizar distância
    candidates = normalize_distance(candidates, latitude, longitude)
    # 3. Calcular avaliação média
    candidates = calculate_average_rating(candidates) # Cria 'media_produtor_produto' e 'avaliacao_norm'

    # Garante que a coluna 'organico' (0 ou 1) exista nos candidatos
    if 'organico' not in candidates.columns:
        candidates['organico'] = 0 # Fallback: assume não orgânico se a coluna estiver ausente
        
    # Prepara os dados para cálculo do score
    features_values_df = pd.DataFrame({
        'avaliacao_norm': candidates['avaliacao_norm'],
        'proximidade': candidates['proximidade'],
        'item_is_organic_actual': candidates['organico'] # Status orgânico real do item
    })
    
    # 4. Calcular score para cada candidato
    candidates["score"] = features_values_df.apply(
        lambda row: calculate_score(0, organic_preference, row.tolist()), axis=1 # Tipo 0 para "Melhores Produtos"
    )

    # 5. Ordenar por score e selecionar os top N (ex: 5)
    top_recommendations = (
        candidates.sort_values(by='score', ascending=False)
        .drop_duplicates(subset=['produto', 'nome_produtor']) # Mantém o melhor score para cada par produto-produtor
        .head(5)
    )

    # Retorna as colunas relevantes, incluindo latitude/longitude para o mapa
    return top_recommendations[[
        'produto', 'nome_produtor', 'local', 'organico', 
        'media_produtor_produto', 'distancia_km', 'score', 'latitude', 'longitude'
    ]].round({'media_produtor_produto': 2, 'distancia_km': 2, 'score': 2})


# Função auxiliar para obter produtores candidatos com base em um produto de interesse
def get_producer_recomendation(df_reviews, product_of_interest):
    """
    Filtra o DataFrame de reviews para encontrar produtores.
    Se 'product_of_interest' for fornecido, retorna produtores que vendem esse produto.
    Caso contrário, retorna todos os produtores distintos presentes nas reviews.
    """
    if df_reviews.empty:
        return pd.DataFrame()
        
    if product_of_interest and product_of_interest.strip() != "":
        # Filtra reviews pelo produto de interesse
        candidates = df_reviews[df_reviews['produto'] == product_of_interest].copy()
    else: # Se nenhum produto específico, considera todos os produtores
        candidates = df_reviews.copy()

    if candidates.empty:
        return pd.DataFrame() 
    
    # Retorna produtores únicos, mantendo a primeira ocorrência de cada um para preservar suas informações
    return candidates.drop_duplicates(subset=['nome_produtor'], keep='first')


# Função para calcular a avaliação média de cada produtor
def calculate_average_producer_rating(candidates_df_from_reviews):
    """
    Calcula a avaliação média geral para cada produtor com base nas reviews fornecidas.
    Agrupa por produtor e outras informações relevantes (local, coordenadas, status orgânico).
    Adiciona 'media_avaliacao' e 'avaliacao_norm' (normalizada de 0 a 1).
    """
    if candidates_df_from_reviews.empty or 'nome_produtor' not in candidates_df_from_reviews.columns:
        return pd.DataFrame()

    grouping_cols = ['nome_produtor', 'local', 'latitude', 'longitude']
    # Assume que 'organico' no nível do produtor significa algo como "certificado" ou "vende produtos orgânicos".
    # Esta coluna deve vir do df_full_reviews e ser mantida em get_producer_recomendation.
    if 'organico' in candidates_df_from_reviews.columns:
        grouping_cols.append('organico')
    else:
        # Se 'organico' estiver ausente, cria um placeholder para não quebrar o agrupamento,
        # assumindo que o produtor não é classificado como orgânico por padrão.
        if not candidates_df_from_reviews.empty:
             candidates_df_from_reviews['organico_placeholder_producer'] = 0 
             if 'organico' not in grouping_cols: grouping_cols.append('organico_placeholder_producer')

    # Agrega as reviews para obter estatísticas por produtor
    producer_agg = (
        candidates_df_from_reviews.groupby(grouping_cols, observed=True, dropna=False)
        .agg(
            media_avaliacao=('avaliacao', 'mean'), # Média das avaliações do produtor
            ).reset_index()
    )
    # Renomeia ou remove a coluna placeholder de orgânico, se usada.
    if 'organico_placeholder_producer' in producer_agg.columns and 'organico' not in producer_agg.columns :
        producer_agg.rename(columns={'organico_placeholder_producer': 'organico'}, inplace=True)
    elif 'organico_placeholder_producer' in producer_agg.columns and 'organico' in producer_agg.columns:
        producer_agg = producer_agg.drop(columns=['organico_placeholder_producer'])


    if producer_agg.empty:
        return pd.DataFrame()

    # Normaliza a avaliação média do produtor (0 a 1)
    producer_agg['avaliacao_norm'] = producer_agg['media_avaliacao'] / 5.0
    return producer_agg


# Função principal para recomendar os "Melhores Produtores"
def recommend_best_productors(product_of_interest, latitude, longitude, organic_preference=0, top_n=5):
    """
    Recomenda os melhores produtores, opcionalmente filtrados por um produto de interesse.
    Classifica os produtores com base em sua avaliação média, proximidade e, potencialmente, status orgânico.
    A 'organic_preference' do usuário não é usada diretamente no score tipo 1 pela função 'calculate_score' atual.
    """
    if df_full_reviews.empty:
        return pd.DataFrame({'mensagem': ['Dataset de reviews não carregado.']})

    # 1. Obter produtores candidatos (que vendem o produto ou todos)
    producers_from_reviews = get_producer_recomendation(df_full_reviews, product_of_interest)

    if producers_from_reviews.empty:
        return pd.DataFrame()

    # 2. Calcular detalhes e avaliação média para esses produtores
    producers_details = calculate_average_producer_rating(producers_from_reviews)

    if producers_details.empty:
        return pd.DataFrame()

    # 3. Calcular e normalizar distância
    producers_details = normalize_distance(producers_details, latitude, longitude)
    
    # Garante que a coluna 'organico' (status do produtor) exista para o score
    if 'organico' not in producers_details.columns:
        producers_details['organico'] = 0 # Fallback: assume não orgânico

    # Prepara os dados para cálculo do score
    features_values_df = pd.DataFrame({
        'avaliacao_norm': producers_details['avaliacao_norm'],
        'proximidade': producers_details['proximidade'],
        'item_is_organic_actual': producers_details['organico'] # Status orgânico do produtor
    })

    # 4. Calcular score para cada produtor
    # Para recomendação tipo 1 (produtores), 'organic_preference' do usuário não é usada pela 'calculate_score'.
    producers_details["score"] = features_values_df.apply(
        lambda row: calculate_score(1, organic_preference, row.tolist()), axis=1 # Tipo 1 para "Melhores Produtores"
    )
    
    # 5. Ordenar por score e selecionar os top N
    top_result = producers_details.sort_values(by='score', ascending=False).head(top_n)

    # Retorna as colunas relevantes
    return top_result[[
        'nome_produtor', 'local', 'organico', 
        'media_avaliacao', 'distancia_km', 'score', 'latitude', 'longitude'
    ]].round({'media_avaliacao': 2, 'distancia_km': 2, 'score': 2})


# Função auxiliar para obter produtos de um produtor específico, excluindo uma lista de indesejados
def get_products_recomendation(df_source_reviews, producer_name, unwanted_products_list):
    """
    Filtra o DataFrame de reviews para encontrar produtos de um 'producer_name' específico.
    Exclui produtos que estão na 'unwanted_products_list'.
    """
    if df_source_reviews.empty or not producer_name:
        return pd.DataFrame()

    if unwanted_products_list is None:
        unwanted_products_list = []

    # Filtra por nome do produtor e remove produtos indesejados
    candidates = df_source_reviews[
        (df_source_reviews['nome_produtor'] == producer_name) &
        (~df_source_reviews['produto'].isin(unwanted_products_list)) # '~' é o operador NOT para boolean Series
    ].copy()
    
    return candidates


# Função principal para recomendar os "Melhores Produtos de um Produtor Específico"
def recommend_best_product_productors(producer_name, local_filter, organic_preference, latitude, longitude, unwanted_products=None):
    """
    Recomenda os melhores produtos de um produtor específico, com opção de filtro por local (RA)
    e preferência por orgânicos.
    """
    if df_full_reviews.empty:
        return pd.DataFrame({'mensagem': ['Dataset de reviews não carregado.']})
    if not producer_name:
         return pd.DataFrame({'mensagem': ['Nome do produtor não fornecido.']})

    # 1. Obter produtos do produtor (excluindo indesejados)
    candidates = get_products_recomendation(df_full_reviews, producer_name, unwanted_products)

    if candidates.empty:
        return pd.DataFrame() 

    # 2. Aplicar filtro de local (Região Administrativa) se fornecido
    if local_filter and not candidates.empty:
        candidates = candidates[candidates['local'] == local_filter].copy()
        if candidates.empty: # Nenhum produto deste produtor na RA especificada
            return pd.DataFrame() 

    if candidates.empty: # Verificação adicional de segurança
        return pd.DataFrame()

    # 3. Calcular e normalizar distância
    candidates = normalize_distance(candidates, latitude, longitude)
    # 4. Calcular avaliação média dos produtos (agrupado por produto e nome_produtor)
    candidates = calculate_average_rating(candidates) 
    
    # Garante que a coluna 'organico' (status do produto) exista
    if 'organico' not in candidates.columns:
        candidates['organico'] = 0 # Fallback

    # Prepara os dados para cálculo do score
    features_values_df = pd.DataFrame({
        'avaliacao_norm': candidates['avaliacao_norm'],
        'proximidade': candidates['proximidade'],
        'item_is_organic_actual': candidates['organico'] # Status orgânico real do produto
    })
    
    # 5. Calcular score para cada produto candidato
    candidates["score"] = features_values_df.apply(
        lambda row: calculate_score(2, organic_preference, row.tolist()), axis=1 # Tipo 2 para "Produtos de Produtor"
    )
    
    # 6. Ordenar por score, remover duplicatas de produto (mantendo o melhor score) e selecionar top N
    resultado = candidates.sort_values(by='score', ascending=False).drop_duplicates(subset=['produto']).head(5)

    # Retorna as colunas relevantes
    return resultado[['produto', 'nome_produtor', 'local', 'organico', 
                      'media_produtor_produto', 'distancia_km', 'score', 'latitude', 'longitude'
                      ]].round({'media_produtor_produto': 2, 'distancia_km': 2, 'score': 2})