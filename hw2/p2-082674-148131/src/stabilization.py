import cv2
import sys
import os

#Arquivo que abre video le imagem por image transforma em cinza e mostra a image.
#Naõ consegui roda-lo no opencv da minha maquina, tive que usar o docker
#Parece que precisa ter almas bibliotaca que não tenho
#Mas como funcionou usando o docker não tem problema.

dir_path = os.path.dirname(os.path.realpath(__file__))
cap = cv2.VideoCapture(dir_path+'/../input/input-p2-2-0-0.mp4')
if cap is None:
	print("Video not found.")
	sys.exit()	
while(cap.isOpened()):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.imshow('frame', gray)
	if cv2.waitKey(1) & 0xFF ==ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
