

from tkinter import *
from PIL import Image, ImageTk
import newFR
import threading as th

def cam_click(but):
    but.destroy()
    t1.start()
    
root = Tk()
root.title('Facial Recognition')
root.geometry('400x200')
 

def check():
     if newFR.quit_ == True:
         print("quited")
         root.destroy()
         root.quit()
     root.after(ms=1,func=check)

    
t1=th.Thread(target = newFR.main , name = 'cam_thread', kwargs=dict(root=root),daemon=True)

root.after(ms=1,func=check)

label=Label(root,text="If you want to train face, press `P`. else press `Q`.")
label.pack()

web_cam = Button(root,text='START')
web_cam['command'] = lambda but=web_cam:cam_click(but)
web_cam.pack()
    
root.mainloop()
#print('x')   
