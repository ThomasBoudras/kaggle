import torch
import base64

# Chemin vers le fichier contenant le modèle
fichier_modele = 'best.pth'

# Charger le modèle
modele = torch.load(fichier_modele, map_location=torch.device('cpu'))

# Accéder aux poids du modèle
poids_du_modele = modele['state_dict']

# Convertir les poids en base64
poids_base64 = base64.b64encode(str(poids_du_modele).encode('utf-8'))

# Afficher les poids en base64
print(poids_base64)
