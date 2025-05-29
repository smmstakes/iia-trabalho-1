import json
import joblib
# import folium # Mantido para mapa inicial, embora o mapa JS seja o principal
import src.recommender_engine as recommender

import pandas as pd

from geopy.geocoders import Nominatim
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Carregar recursos essenciais para a aplicação
try:
    resources = joblib.load('./data/model/full_resources.pkl') # Dicionário com listas de produtos e produtores formatados
    df_full_reviews = pd.read_parquet('./data/datasets/df_full_reviews.parquet') # DataFrame com todas as avaliações
    
    # Disponibiliza df_full_reviews para o módulo recommender, se ainda não estiver carregado lá
    if recommender.df_full_reviews is None or recommender.df_full_reviews.empty:
        recommender.df_full_reviews = df_full_reviews
    # Disponibiliza resources para o módulo recommender, se ainda não estiver carregado lá
    if recommender.resources is None:
        recommender.resources = resources
except FileNotFoundError:
    print("Erro ao carregar recursos. Verifique os caminhos dos arquivos: ./data/model/full_resources.pkl ou ./data/datasets/df_full_reviews.parquet")
    resources = {'products_list': [], 'producers_formatted': []} 
    df_full_reviews = pd.DataFrame()
    if recommender.df_full_reviews is None: recommender.df_full_reviews = pd.DataFrame()
    if recommender.resources is None: recommender.resources = {}
except Exception as e:
    print(f"Erro ao carregar recursos: {e}")
    resources = {'products_list': [], 'producers_formatted': []}
    df_full_reviews = pd.DataFrame()
    if recommender.df_full_reviews is None: recommender.df_full_reviews = pd.DataFrame()
    if recommender.resources is None: recommender.resources = {}


# Rota principal da aplicação
@app.route('/')
def index():
    # O mapa Folium inicial foi comentado, pois o mapa Leaflet.js é o principal e mais dinâmico.
    # initial_map_folium = folium.Map(
    # location=[-15.7942, -47.8822],
    # zoom_start=5,
    # tiles='OpenStreetMap'
    # )

    producers_ra_data = {} # Dicionário para armazenar dados de produtores por Região Administrativa
    try:
        # Carrega dados JSON que associam produtores às suas Regiões Administrativas
        with open('./data/json/producers_ra.json', 'r') as f: 
            producers_ra_data = json.load(f) 
    except FileNotFoundError:
        print("Arquivo './data/json/producers_ra.json' não encontrado.")
    except json.JSONDecodeError:
        print("Erro ao decodificar o JSON em './data/json/producers_ra.json'.")

    # Renderiza o template principal, passando listas de produtos, produtores e o JSON de produtores/RAs
    return render_template('index.html',
                           initial_map="", # Placeholder para o mapa, já que o Leaflet.js cuidará disso
                           products_list=resources.get('products_list', []), 
                           producers_list=list(producers_ra_data.keys()), 
                           producers_ra_json=json.dumps(producers_ra_data) 
                           )


# Rota para obter a localização do usuário via IP (fallback caso a geolocalização do navegador falhe)
@app.route('/get_location')
def get_location():
    geolocator = Nominatim(user_agent="geoapi_local_products_v2") # Serviço de geocodificação
    try:
        ip_address = request.headers.get('X-Forwarded-For', request.remote_addr) # Obtém o IP do usuário
        # Se for um IP local ou inválido, retorna a localização padrão (Brasília)
        if ip_address == '127.0.0.1' or not ip_address: 
            print("IP Local/Inválido detectado, usando Brasília como fallback para geocodificação por IP.")
            return jsonify({'latitude': -15.7942, 'longitude': -47.8822, 'message': 'IP Local, fallback para Brasília'})

        location = geolocator.geocode(ip_address) # Tenta obter a localização a partir do IP
        if location:
            return jsonify({
                'latitude': location.latitude,
                'longitude': location.longitude
            })
    except Exception as e:
        print(f"Erro em get_location: {e}")
    # Fallback final para Brasília em caso de erro
    return jsonify({'latitude': -15.7942, 'longitude': -47.8822, 'message': 'Fallback Brasília'})

# Rota para processar os pedidos de recomendação
@app.route('/recommend', methods=['POST'])
def handle_recommendation():
    data = request.json # Obtém os dados enviados pelo frontend
    rec_type = data.get('type') # Tipo de recomendação solicitada
    result = pd.DataFrame()

    # Garante que o motor de recomendação tenha acesso aos dados necessários
    if (recommender.df_full_reviews is None or recommender.df_full_reviews.empty) and \
       (not df_full_reviews.empty):
        recommender.df_full_reviews = df_full_reviews
    if (recommender.resources is None) and resources:
        recommender.resources = resources

    # Verifica se o dataset de reviews está carregado; essencial para as recomendações
    if recommender.df_full_reviews is None or recommender.df_full_reviews.empty:
        return jsonify({'error': 'Dataset de reviews não carregado no servidor.'}), 500

    try:
        # Obtém dados de geolocalização e preferência por orgânicos do request
        latitude = float(data.get('latitude', -15.7942)) # Padrão para Brasília
        longitude = float(data.get('longitude', -47.8822)) # Padrão para Brasília
        organic_preference = int(data.get('organic', 0)) # 0 para "Não", 1 para "Sim"

        # Lógica de recomendação baseada no tipo ('products', 'producers', 'producer-products')
        if rec_type == 'products':
            selected_locations = data.get('locations', [])
            location_filter = selected_locations[0] if selected_locations else None

            # Chama a função do motor de recomendação para melhores produtos
            result = recommender.recommend_best_products(
                desired_products=data.get('products', []),
                producer=data.get('producer'),
                location=location_filter, 
                organic_preference=organic_preference,
                latitude=latitude,
                longitude=longitude
            )
        elif rec_type == "producers":
            # Chama a função do motor de recomendação para melhores produtores
            result = recommender.recommend_best_productors(
                product_of_interest=data.get('single_product', ''),
                latitude=latitude,
                longitude=longitude,
                organic_preference=organic_preference # Passa a preferência, embora o motor possa não usá-la diretamente para este tipo
            )
        elif rec_type == "producer-products":
            selected_locations = data.get('locations', [])
            local_filter = selected_locations[0] if selected_locations else None
            producer_name = data.get('producer')

            if not producer_name:
                 return jsonify({'error': 'Produtor não especificado para "Melhores Produtos do Produtor".'}), 400
            
            # Chama a função do motor de recomendação para melhores produtos de um produtor específico
            result = recommender.recommend_best_product_productors(
                producer_name=producer_name, 
                local_filter=local_filter, 
                organic_preference=organic_preference,
                latitude=latitude,
                longitude=longitude,
                unwanted_products=data.get('unwanted_products', []) 
            )
        else:
            return jsonify({'error': 'Tipo de recomendação inválido'}), 400

        # Se nenhum resultado for encontrado, retorna uma lista vazia para o frontend tratar
        if result is None or result.empty: 
            return jsonify([]) 
        
        # Garante que as colunas de latitude e longitude estejam nos resultados para plotagem no mapa.
        # O motor de recomendação já deve incluir essas colunas.
        if 'latitude' not in result.columns or 'longitude' not in result.columns:
            pass

        # Retorna os resultados da recomendação em formato JSON
        return jsonify(result.to_dict(orient='records'))

    except Exception as e:
        # Tratamento de exceções durante o processo de recomendação
        print(f"Erro durante a recomendação: {e}")
        import traceback
        traceback.print_exc() # Imprime o stack trace para debugging
        return jsonify({'error': f'Ocorreu um erro no servidor: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True) # Executa a aplicação Flask em modo debug