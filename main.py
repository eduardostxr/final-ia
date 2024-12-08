import cv2
import numpy as np

TINY = False
ARQUIVO_CFG = f"yolov3{'-tiny' if TINY else ''}.cfg"
ARQUIVO_PESOS = f"yolov3{'-tiny' if TINY else ''}.weights"
ARQUIVO_CLASSES = f"coco{'-tiny' if TINY else ''}.names"

with open(ARQUIVO_CLASSES, "r") as f:
    CLASSES = [linha.strip() for linha in f.readlines()]
CORES = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def inicializar_detector_de_faces():
    classificador = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if classificador.empty():
        raise IOError("Não foi possível carregar o modelo de detecção de faces.")
    return classificador

def carregar_modelo_pretreinado():
    modelo = cv2.dnn.readNetFromDarknet(ARQUIVO_CFG, ARQUIVO_PESOS)
    modelo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    modelo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    if modelo.empty():
        raise IOError("Não foi possível carregar o modelo de detecção de objetos.")
    return modelo

def detectar_objetos(frame, modelo, limiar=0.5):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True)
    modelo.setInput(blob)
    camadas_saida = [modelo.getLayerNames()[i - 1] for i in modelo.getUnconnectedOutLayers()]
    saidas = modelo.forward(camadas_saida)

    objetos_detectados = []
    altura, largura = frame.shape[:2]
    caixas, confiancas, ids_classes = [], [], []

    for saida in saidas:
        for deteccao in saida:
            pontuacoes = deteccao[5:]
            id_classe = np.argmax(pontuacoes)
            confianca = pontuacoes[id_classe]
            if confianca > limiar:
                x, y, largura_caixa, altura_caixa = (deteccao[0:4] * np.array([largura, altura, largura, altura])).astype("int")
                caixas.append([x - largura_caixa//2, y - altura_caixa//2, largura_caixa, altura_caixa])
                confiancas.append(float(confianca))
                ids_classes.append(id_classe)

    indices = cv2.dnn.NMSBoxes(caixas, confiancas, limiar, limiar - 0.1)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, largura_caixa, altura_caixa = caixas[i]
            cor = [int(c) for c in CORES[ids_classes[i]]]
            cv2.rectangle(frame, (x, y), (x + largura_caixa, y + altura_caixa), cor, 2)
            cv2.putText(frame, f"{CLASSES[ids_classes[i]]}: {confiancas[i]:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
            objetos_detectados.append(CLASSES[ids_classes[i]])

    return objetos_detectados

def desenhar_faces(frame, faces):
    for (x, y, largura, altura) in faces:
        cv2.rectangle(frame, (x, y), (x + largura, y + altura), (245, 255, 0), 2)

def main():
    classificador_de_faces = inicializar_detector_de_faces()
    modelo = carregar_modelo_pretreinado()
    captura_video = cv2.VideoCapture(0)

    if not captura_video.isOpened():
        raise Exception("Não foi possível abrir a webcam.")

    captura_video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    captura_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    limiar_confianca = 0.5
    cv2.namedWindow('Detecta Objetos e Faces')
    if TINY:
        cv2.createTrackbar('Limiar de Confiança', 'Detecta Objetos e Faces', int(limiar_confianca * 100), 100, lambda v: None)

    while True:
        ret, frame = captura_video.read()
        if not ret:
            break

        objetos_detectados = detectar_objetos(frame, modelo, limiar_confianca)
        faces = classificador_de_faces.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(30, 30))

        desenhar_faces(frame, faces)

        if "celular" in objetos_detectados and len(faces) > 0:
            cv2.putText(frame, "Usando celular", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Detecta Objetos e Faces', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    captura_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()