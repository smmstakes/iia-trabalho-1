from django.db import models

class Producer(models.Model):
    name = models.CharField(max_length=200)
    latitude = models.FloatField()
    longitude = models.FloatField()
    products = models.TextField()  # Lista de produtos separados por vírgula
    is_organic = models.BooleanField(default=False)

    def __str__(self):
        return self.name

class UserProfile(models.Model):
    name = models.CharField(max_length=100)
    latitude = models.FloatField()
    longitude = models.FloatField()
    preferred_products = models.TextField()  # Lista separada por vírgula
    prefers_organic = models.BooleanField(default=False)
    max_distance_km = models.FloatField(default=20)

    def __str__(self):
        return self.name

class Review(models.Model):
    producer = models.ForeignKey(Producer, on_delete=models.CASCADE)
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    rating = models.IntegerField()  # 1 a 5
    comment = models.TextField(blank=True)

    def __str__(self):
        return f"{self.user.name} -> {self.producer.name} ({self.rating})"