from kivy.app import App
from kivy.uix.button import Button
from kivy.metrics import dp
import numpy as np
import cv2
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from kivy.uix.floatlayout import FloatLayoutq

# charger un modele pre-entrainer
model = ResNet50(weights='imagenet')

# Fonction pour predire si l'image contient un dog
def dog_detector(img):
    img = cv2.resize(img, (224, 224))  # redimensionner l'image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir  l'image to RGB
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    prediction = model.predict(img)
    return (np.argmax(prediction) <= 268) and (np.argmax(prediction) >= 151)

# fonction pour activer la camera et detection en temps reel
def live_dog_detection(instance):
    cap = cv2.VideoCapture(0)  # utiliser la 1er camera (0) - adjuster si necessaire

    while True:
        ret, frame = cap.read()  # Capture a partir de la camera

        # Detection de chien en temps reel
        if dog_detector(frame):
            text = "Chien detecte"
        else:
            text = "Pas de chien detecte"

        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Dog Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

class MonApplication(App):
    
    def __init__(self, **kwargs):
        super(MonApplication, self).__init__(**kwargs)

    def build(self):
        layout = FloatLayout()
        
        button = Button(text="Détecter le chien", size_hint=(None, None), size=(dp(200), dp(100)))
        button.bind(on_press=live_dog_detection)  # Cliquez sur le bouton pour executer la fonction
        button.pos_hint = {'center_x': 0.5, 'center_y': 0.5}  # Centrer le boutton
        layout.add_widget(button)
        return layout

if __name__ == "__main__":
    MonApplication().run()
