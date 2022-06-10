import cv2
import numpy as np

filename = 'img/z.png'

img_gray = cv2.imread(filename, 0)

lastimage=np.full((len(img_gray),len(img_gray[0])),255,"uint8")

pixel=5*2

print(len(img_gray[0]),len(img_gray))

x=[]
y=[]
for i in range(len(img_gray[0])):
	if i%pixel!=0:
		continue
	cnt=0
	for j in range(len(img_gray)):
		if cnt>0:
			cnt=cnt-1
			continue
		if img_gray[j][i]<100:
			cnt = pixel-1
			cv2.circle(lastimage, (i, j), 1, 0, thickness=-1)
			x.append(float(i))
			y.append(float(j))
cv2.imshow("soblex+y",lastimage)


if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

print(x)
print(y)
print(len(x))
print(len(y))
