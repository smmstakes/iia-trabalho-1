import pandas as pd
from geopy.distance import geodesic
from .models import Producer, UserProfile, Review

def calculate_distance_km(coord1, coord2):
    return geodesic(coord1, coord2).km

def get_recommendations_for_user(user: UserProfile):
    user_coords = (user.latitude, user.longitude)
    preferred = set(p.strip().lower() for p in user.preferred_products.split(','))
    producers = Producer.objects.all()

    results = []

    for producer in producers:
        prod_coords = (producer.latitude, producer.longitude)
        distance = calculate_distance_km(user_coords, prod_coords)

        if distance > user.max_distance_km:
            continue

        if user.prefers_organic and not producer.is_organic:
            continue

        producer_products = set(p.strip().lower() for p in producer.products.split(','))
        matching = preferred.intersection(producer_products)

        if not matching:
            continue

        # Average rating
        reviews = Review.objects.filter(producer=producer)
        if reviews.exists():
            avg_rating = sum([r.rating for r in reviews]) / reviews.count()
        else:
            avg_rating = 0

        results.append({
            'producer': producer,
            'distance_km': round(distance, 2),
            'matching_products': list(matching),
            'avg_rating': round(avg_rating, 2)
        })

    results.sort(key=lambda x: (x['distance_km'], -x['avg_rating']))
    return results