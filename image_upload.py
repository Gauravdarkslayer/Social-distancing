import os
import urllib.request
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, jsonify

UPLOAD_FOLDER = './UPLOADS'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg','mp4','avi'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/file-upload', methods=['POST'])
def upload_file():
	# check if the post request has the file part
	if 'file' not in request.files:
		resp = jsonify({'message' : 'No video part in the request'})
		resp.status_code = 400
		return resp
	file = request.files['file']
	if file.filename == '':
		resp = jsonify({'message' : 'No video selected for uploading'})
		resp.status_code = 400
		return resp
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		resp = jsonify({'message' : 'video successfully uploaded'})
		resp.status_code = 201

		prediction = predict_distance(filename)
		return prediction
		# status = jsonify({"Message":"sucess"})
		# return status
	else:
		resp = jsonify({'message' : 'Allowed video types are .mp4,.avi'})
		resp.status_code = 400
		return resp
		
@app.route('/status',methods=['GET'])
def success ():
	# status = jsonify({"Message":"sucessyooooooooooooooooooo"})
	return return_message

return_message=""

def predict_distance(filename):
		global return_message
		# import the necessary packages
		import numpy as np
		import cv2
		import math
		
			
		# Load Yolo
		net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
		#classes=['Person','Car']
		classes = []
		with open("coco.names", "r") as f:
			classes = [line.strip() for line in f.readlines()]
			#person_ind = [i for i, cls in enumerate(net.classes) if cls == 'person'][0]
		layer_names = net.getLayerNames()
		output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
		#colors = np.random.uniform(0, 255, size=(len(classes), 3))


		cap = cv2.VideoCapture(filename)

		count =0
		count_pic=0
		count_off=0
		d=0
		while cap.isOpened():
			# Capture frame-by-frame
			ret, img = cap.read()

			height, width, channels = img.shape

			# Detecting objects
			blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

			net.setInput(blob)
			outs = net.forward(output_layers)

			# Showing informations on the screen
			class_ids = []
			confidences = []
			boxes = []
			
			
			for out in outs:
				for detection in out:
					scores = detection[5:]
					class_id = np.argmax(scores)
					confidence = scores[class_id]
					if confidence > 0.5:
						# Object detected
						center_x = int(detection[0] * width)
						center_y = int(detection[1] * height)
						w = int(detection[2] * width)
						h = int(detection[3] * height)

						# Rectangle coordinates
						x = int(center_x - w / 2)
						y = int(center_y - h / 2)

						boxes.append([x, y, w, h])
						confidences.append(float(confidence))
						class_ids.append(class_id)

			indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
			#print(indexes)
			font = cv2.FONT_HERSHEY_PLAIN
			
			count_ppl=0
			l=[]
			l=[]
			lf=[]
			j=1
			
			for i in range(len(boxes)):
				if i in indexes:
					x, y, w, h = boxes[i]
					label = str(classes[class_ids[i]])
					#color = colors[i]
					count_ppl+=1
					if label=='person':
						cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
						l=[]
						l.append(x)
						l.append(y)
						lf.append(l)
					
					s=str(j)
					j+=1
					
			close_person=""
			off=0
			
			for i in range(len(lf)):
				for j in range(i+1,len(lf)):
					d= ((lf[j][1]-lf[i][1])*2)+((lf[j][0]-lf[i][0])*2)
					if d>0:
						d=math.sqrt(d)
					else:
						continue
					#print("P",i+1,"- P",j+1,"=",d)
					if d<65:
						close_person+="Person "+str(i+1)+" and Person "+str(j+1)+" ; "
						#if label=='person':
						img = cv2.line(img, (lf[i][0]+15,lf[i][1]+35), (lf[j][0]+15,lf[j][1]+35), (0,0,255), 2)
						off+=1
					
					
			count+=1    
			if count>=15:
				print("FRAME "+str(count_pic)+"    People Count : "+str(count_ppl)+"   Offenders : "+str(off))
				# cv2.imwrite('dataset\\img'+str(count_pic)+'.png',img)
				count_pic+=1
				if off>5:
					# cv2.imwrite('offenders\\img'+str(count_off)+'.png',img)
					count_off+=1
					
			
			if count_ppl>=10:
				a="HIGH ALERT "+str(count_ppl)+"people in your area!"
				# print(a) 
				status = jsonify({"Message":a})
				return_message=status
				return status   
			
			
			if count>=15:
				count=0 
				off=0
			
			
			# cv2.imshow('frame',img)
			# if cv2.waitKey(1) & 0xFF == ord('q'):
			# 	break

		# When everything done, release the capture
		# cap.release()
		# and release the output
		#out.release()
		# finally, close the window
		# cv2.destroyAllWindows()
		# cv2.waitKey(1)
		
		status = jsonify({"Message":close_person})
		return_message=status
		return status

if __name__ == "__main__":
    app.run(host="127.0.0.1",port=80)