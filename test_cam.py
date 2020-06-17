import numpy as np
import cv2
cap = cv2.VideoCapture(0) #参数为0时调用本地摄像头；url连接调取网络摄像头；文件地址获取本地视频
# while(True):
#     ret,frame=cap.read()

# #灰度化
#     # gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     # cv2.imshow('frame',gray)

#     #普通图片
#     cv2.imshow('frame',frame)

#     if cv2.waitKey(1)&0xFF==ord('q'):
#         break

ret,frame=cap.read()
a = np.array(frame)
print(a.shape)

cap.release()
cv2.destroyAllWindows()