# -*- coding: utf-8 -*-

import time
import os
import logging
import numpy
import cv2
import pickle
from tkinter import *
import threading as th
import sys
from PIL import ImageTk


wait_for_input=True
exit_flag=False
name=None
created=False
quit_=False
exited=False

log = logging.getLogger()
log.setLevel(logging.INFO)

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
log.addHandler(sh)


VIDEO_DEVICE_INT = 0
LBPH_LABEL_NUM = 0

#def get_max_label(face_recognizer):
#    train_face_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'individual_face.xml')
#    try:
#        face_recognizer.read(train_face_filepath)
#        return numpy.asscalar(max(face_recognizer.getLabels()))
#    except :
#        return -1

def _get_color_and_gray_frame_helper(capture_device):
        global quit_
        ret , frame = capture_device.read()
        if ret is True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame, gray
        else:
            print("Something wrong happened!!!")
            quit_=True
#            sys.exit(1)

def _detect_faces_helper(face_classifier, gray_frame):
    """
    Using shameful helper to make universal changes to the 
    detect method
    """
    height, width = gray_frame.shape
    min_size = (int(height/4), int(width/4)) 
    return face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=8, flags=cv2.CASCADE_SCALE_IMAGE, minSize=min_size)


def reset_globals():
    global exited
    global wait_for_input
    global exit_flag
    global name
    global created
    global quit_
    
    exited=False
    quit_=False
    created=False
    wait_for_input=True
    exit_flag=False
    name=None
    


def get_name(strName,but,ent,can,root):
    global wait_for_input
    global name
    
    name=strName.get()
    wait_for_input=False
    
    but.destroy()
    ent.destroy()
    can.destroy()
    
    create_exit_but(root)
    
def start_again(but1,but2,root):

    global exit_flag
    exit_flag=True
    but1.destroy()
    but2.destroy()
# =============================================================================
#     t1=None
#     for x in th.enumerate() :
#          if x.getName()=='cam_thread':
#              t1=x
#      
#     try:
#          t1.join()
#     except:
#          pass
# =============================================================================
    
                
    while True:
        if exited==True:    
            t1=th.Thread(target = main , name = 'cam_thread', kwargs=dict(root=root),daemon=True)
            t1.start()
            break

def exit_(root):
    global exit_flag
    global quit_
#    exit_flag=True
# =============================================================================
#     t1=None
#     for x in th.enumerate() :
#         if x.getName()=='cam_thread':
#             t1=x
#     
#     try:
#         t1.join()
#     except:
#         pass
# =============================================================================
    reset_globals()
    quit_=True
    #sys.exit(0)
    #root.quit()
    
def create_exit_but(root):
    
    global created
    created=True
    
    exit_but=Button(root,text='Exit')
    exit_but['command'] = lambda root=root :exit_(root)
    exit_but.pack()
    
    start_again_but = Button(root,text='Start Again')
    start_again_but['command'] = lambda but1=start_again_but,but2=exit_but,root=root:start_again(but1,but2,root)
    start_again_but.pack()
    


def main(root = None):
    
    global wait_for_input
    global name
    global exit_flag
    global created
    global VIDEO_DEVICE_INT
    global exited
    
    reset_globals()
    
    haarcascade_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'haarcascade_frontalface_default.xml')

    face_classifier = cv2.CascadeClassifier(haarcascade_filepath)
    
    # should really test here and see if this number works
    print('Using device number: {}'.format(VIDEO_DEVICE_INT))
    capture_device = cv2.VideoCapture(VIDEO_DEVICE_INT)
    
    train_face = False 
    print("If you want to train face, press `P`. else press `Q`")

    # This is the best loop ever! So much control
    while True:
        frame, gray = _get_color_and_gray_frame_helper(capture_device)
        faces = _detect_faces_helper(face_classifier, gray) 

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('YOUR FACE', frame)

        c = cv2.waitKey(2) & 0xFF

        if c == ord('q'):
            break
        elif c == ord('p'):
            train_face = True
            break
    
    cv2.destroyAllWindows()
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    print(face_recognizer.getLabels())
    # If training for face
    if train_face:
        
        images = []
        cur_id=0
        labels = []
        names = {}
        #print(labels)
        
        try:
           with open('Temp\\names.pickle','rb') as f:
               names = pickle.load(f)
        except:
            names= {}
        
        try:
           with open('Temp\\new_images.pickle','rb') as f:
               images = pickle.load(f)
        except:
            images= []
        try:    
            with open('Temp\\labels.pickle','rb') as f:
                labels = pickle.load(f)
        except:
            labels=[]
        try:    
            with open('Temp\\cur_id.pickle','rb') as f:
                cur_id = pickle.load(f)
        except:
             cur_id=0
        
        can=Canvas(root,width=200,height=200)
        while(True):
            # don't really need the color frame here, just get gray
            color, gray = _get_color_and_gray_frame_helper(capture_device)
            # check for faces
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            RGB_color=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # since we're assuming one face, if more assume bad data
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_region_of_interest = gray[x:x+w, y:y+h]
                roi_color=RGB_color[x:x+w,y:y+h]
# =============================================================================
#                 try:
#                      img=ImageTk.PhotoImage(roi_color)
#                      can.create_image(x=0,y=0,image=img)
#                      can.pack()
#                 except:
#                      print('Failed')
#                      sys.exit(1)
# =============================================================================
                images.append(face_region_of_interest)
                break

            
            
        strName=StringVar(root)
        Ename=Entry(root,width=25 , textvariable = strName)
        Ename.insert(0,"Name")
        Ename.pack()
        
        train_but=Button(root,text='Train' , command = get_name)
        train_but['command']=lambda strName=strName,but=train_but,ent=Ename,can=can,root=root : get_name(strName,but,ent,can,root)
        train_but.pack()
        
        #print("waiting")
        while(wait_for_input):
            pass
        #print("Resumed")
        #print(name)
        #time.sleep(5)
        #sys.exit(1)
        
        if len(name) > 0:
            
            
            if not name in names.values():
                names[cur_id]=name
                labels.append(cur_id)
                cur_id+=1
            else:
                new_names={v:k for k,v in names.items()}
                labels.append(new_names[name])
                #print(new_names[strName])
                 
            with open('Temp\\names.pickle','wb') as f:
                pickle.dump(names,f)
        
            with open('Temp\\new_images.pickle','wb') as f:
                pickle.dump(images,f)
                
            with open('Temp\\labels.pickle','wb') as f:
                pickle.dump(labels,f)
                
            with open('Temp\\cur_id.pickle','wb') as f:
                pickle.dump(cur_id,f)
             
#            print("Training for Yo Face! ")
#            print(images)
#            print(labels)
            face_recognizer.train(numpy.array(images), numpy.array(labels))
            face_recognizer.save('Temp\\individual_face.xml')
        
    if not train_face:
        
        train_face_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Temp\\individual_face.xml')
        #log.debug('face filepath', train_face_filepath)
        """
        if os.getenv("TRAINED_FACE"):
            print('Loading trained face from environmental variable')
            train_face_filepath = os.getenv("TRAINED_FACE")
        """
        face_recognizer.read(train_face_filepath)

    if not created:
        create_exit_but(root)
    print("Identifying your face!")
    #print(face_recognizer.getLabels())
    precision = 0
    while True:
#        print("Test")
        color, gray = _get_color_and_gray_frame_helper(capture_device)
        faces = _detect_faces_helper(face_classifier, gray)

#        log.debug('detected faces in Identify face', faces)
        
        """labels = []
        try:    
            with open('Temp\\labels.pickle','rb') as f:
                labels = pickle.load(f)
        except:
            labels = []
            
        print(labels)"""
        
        try:
            
            names = {}
        
            try:
                with open('Temp\\names.pickle','rb') as f:
                    names = pickle.load(f)
            except:
                    names= {}
            
            
            for (x, y, w, h) in faces:
                cv2.rectangle(color, (x,y), (x+w, y+h), (255, 0, 0), 2)
                face_region_of_interest = gray[x:x+w, y:y+h]
                label , precision = face_recognizer.predict(face_region_of_interest)
                #print(precision)
                if len(names)>0:
                    if precision>4 and precision<60 :
                        cv2.putText(color,str(names[label]),(x,y-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                    else: 
                        cv2.putText(color,'Recognizing',(x,y-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                    
                #print('Estimated Trained Face : {}, {}, {}'.format(precision, label, w))
            
            #cv2.putText(color,str(precision),(x+40,y-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            #if precision
            
            cv2.imshow('FACE', color)
            
        except cv2.error as er:
            print(er)
#        if cv2.waitKey(3):
#            pass

        if cv2.waitKey(3) and quit_==True:
#             print('In quit if')
             capture_device.release()
             cv2.destroyAllWindows()
             break
 #            root.destroy()
        if cv2.waitKey(3) and exit_flag==True:
#             print('In exit if')
             capture_device.release()
             cv2.destroyAllWindows()
             break

    exited=True
#    print("Exited")
#    capture_device.release()
#    cv2.destroyAllWindows()
#if __name__ == '__main__':
#    main()
