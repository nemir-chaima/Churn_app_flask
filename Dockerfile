# syntax=docker/dockerfile:1
FROM python:3.12.4-slim as base


#specify workdir
WORKDIR /app

#dependencies 
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copier les rest
COPY . .

# Exposer le port 5000 
EXPOSE 5000

# Commande pour lancer l'application Flask
CMD ["python", "app.py"]


