#%%
import numpy as np 
import pandas as pd 
import cv2 
#%%
data=pd.read_csv('./testPic/A6.csv')
img=cv2.imread('./testPic/A6.jpg')
room=cv2.imread('B.jpg')
h,w,d=img.shape
print('Shape of the photo',img.shape[1],room.shape[1])
rio=1000
print('The riot of line',rio)
#%%
for i in range(len(data)):
    confidence=data.iloc[i,1]
    print('Confidence',confidence)
    l_gaze_x=data.iloc[i,2]
    l_gaze_y=data.iloc[i,3]
    l_gaze_z=data.iloc[i,4]
    print('The vector of  left gaze',l_gaze_x,l_gaze_y,l_gaze_z)
    r_gaze_x=data.iloc[i,5]
    r_gaze_y=data.iloc[i,6]
    r_gaze_z=data.iloc[i,7]
    print('The vector of right gaze',r_gaze_x,r_gaze_y,r_gaze_z)
    avg_gaze_x=l_gaze_x+r_gaze_x
    avg_gaze_y=l_gaze_y+r_gaze_y
    l_center_x=int((data.iloc[i,33]+data.iloc[i,37])/2)
    l_center_y=int((data.iloc[i,88]+data.iloc[i,93])/2)
    l_center_X=(data.iloc[i,145]+data.iloc[i,149])/2
    l_center_Y=(data.iloc[i,201]+data.iloc[i,205])/2
    l_center_Z=(data.iloc[i,257]+data.iloc[i,261])/2
    l_dest_m=(l_center_X+rio*l_gaze_x,l_center_Y+rio*l_gaze_y,l_center_Z+rio*l_gaze_z)
    print('Two points in m of left eye',(l_center_X,l_center_Y,l_center_Z),l_dest_m)
    print('The center of left eye in pixel',l_center_x,l_center_y)
    r_center_x=int((data.iloc[i,61]+data.iloc[i,65])/2)
    r_center_y=int((data.iloc[i,117]+data.iloc[i,121])/2)
    r_center_X=(data.iloc[i,173]+data.iloc[i,177])/2
    r_center_Y=(data.iloc[i,229]+data.iloc[i,233])/2
    r_center_Z=(data.iloc[i,285]+data.iloc[i,289])/2
    r_dest_m=(r_center_X+rio*r_gaze_x,r_center_Y+rio*r_gaze_y,r_center_Z+rio*r_gaze_z)
    print('Two points in m of left eye',(r_center_X,r_center_Y,r_center_Z),r_dest_m)
    eye_center_x=int(data.iloc[i,323])
    eye_center_y=int(data.iloc[i,391])
    eye_center_X=data.iloc[i,460]
    eye_center_Y=data.iloc[i,528]
    eye_center_Z=data.iloc[i,596]
    mouth_center_x=int(data.iloc[i,358])
    mouth_center_y=int(data.iloc[i,426])
    gaze_v_len=((eye_center_x-mouth_center_x)**2+(eye_center_y-mouth_center_y)**2)**0.5
    l_most_x=int(data.iloc[i,18])
    l_most_y=int(data.iloc[i,74])
    r_most_x=int(data.iloc[i,52])
    r_most_y=int(data.iloc[i,108])
    pitch=data.iloc[i,293]
    yaw=data.iloc[i,294]
    roll=data.iloc[i,295]
    Rx=np.array([[1,0,0],[0,np.cos(roll),np.sin(roll)],[0,-np.sin(roll),np.cos(roll)]])
    Ry=np.array([[np.cos(pitch),0,-np.sin(pitch)],[0,1,0],[np.sin(pitch),0,np.cos(pitch)]])
    Rz=np.array([[np.cos(yaw),np.sin(yaw),0],[-np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    Mw=np.dot(np.dot(Rx,Ry),Rz)
    init_gaze=np.array([0,0,-1])
    #changed_gaze=np.dot(init_gaze,Mw)
    changed_gaze=np.dot(np.linalg.inv(Mw),init_gaze)
    ###
    l_center=(l_center_x,l_center_y)
    r_center=(r_center_x,r_center_y)
    #eye_center=(eye_center_x,eye_center_y)
    l_center_r=(int(w-l_center_x),int(l_center_y))
    r_center_r=(int(w-r_center_x),int(r_center_y))
    l_gaze_x_r=-l_gaze_x
    l_gaze_y_r=l_gaze_y
    r_gaze_x_r=-r_gaze_x
    r_gaze_y_r=r_gaze_y
    k1=l_gaze_y_r/l_gaze_x_r
    k2=r_gaze_y_r/r_gaze_x_r
    b1=l_center_r[1]-k1*l_center_r[0]
    b2=r_center_r[1]-k2*r_center_r[0]
    intersection=(int((b2-b1)/(k1-k2)),int((k1*b2-k2*b1)/(k1-k2)))
    #eye_dest_r=(int((w-eye_center_x)+rio*(-l_gaze_x)),int(eye_center_y+rio*l_gaze_y))
    eye_dest_l_r=(int((w-l_center_x)+rio*(-l_gaze_x)),int(l_center_y+rio*l_gaze_y))
    eye_dest_r_r=(int((w-r_center_x)+rio*(-r_gaze_x)),int(r_center_y+rio*r_gaze_y))
    #print(l_gaze_x,l_gaze_y)
    #print('The start point of eye after transformation',eye_center_l_r,eye_dest_l_r)
    cv2.line(room,l_center_r,eye_dest_l_r,(0,255,0),2,8)
    cv2.line(room,r_center_r,eye_dest_r_r,(0,255,0),2,8)
    cv2.circle(room,intersection,100,(0,255,0),5,0)
    #cv2.line(room,(0,0),(100,100),(0,255,0),2,8)
    #cv2.line(room,(2475,1354),(3440,1580),(0,255,0),2,8)
    '''
    l_most=(l_most_x,l_most_y)
    r_most=(r_most_x,r_most_y)        
    #r_dest=(int(r_center_x+500*r_gaze_x),int(r_center_y+500*r_gaze_y))
    ###
    if yaw>=0.30 and yaw <=0.50 :
        gaze_x=-(eye_center_y-mouth_center_y)/gaze_v_len
        gaze_y=(eye_center_x-mouth_center_x)/gaze_v_len
        lm_dest=(int(l_most_x+50*gaze_x),int(l_most_y+50*gaze_y))
        cv2.line(img,l_most,lm_dest,(0,255,0),2,8)    
    elif yaw>=-0.50 and yaw<=-0.30:
        gaze_x=(eye_center_y-mouth_center_y)/gaze_v_len
        gaze_y=-(eye_center_x-mouth_center_x)/gaze_v_len
        rm_dest=(int(r_most_x+50*gaze_x),int(r_most_y+50*gaze_y))
        cv2.line(img,r_most,rm_dest,(0,255,0),2,8)    
    elif yaw>-0.30 and yaw<0.30:
        eye_dest=(int(eye_center_x+50*changed_gaze[0]),int(eye_center_y+50*changed_gaze[1]))
        #eye_dest=(int(eye_center_x+50*avg_gaze_x),int(eye_center_y+50*avg_gaze_y))
        cv2.line(img,eye_center,eye_dest,(0,255,0),2,8)
    #cv2.line(img,l_center,l_dest,(0,255,0),2,8)
    #cv2.line(img,r_center,r_dest,(0,255,0),2,8)
    '''
#%%
cv2.imwrite('./room/A6r.jpg',room)
cv2.imshow('img',room)
cv2.waitKey()
cv2.destroyAllWindows()