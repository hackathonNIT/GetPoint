import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import sys

sys.setrecursionlimit(100000)

class UnionFind():
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.parents[x] > self.parents[y]:
            x, y = y, x

        self.parents[x] += self.parents[y]
        self.parents[y] = x

    def size(self, x):
        return -self.parents[self.find(x)]

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def members(self, x):
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    def roots(self):
        return [i for i, x in enumerate(self.parents) if x < 0]

    def group_count(self):
        return len(self.roots())

    def all_group_members(self):
        group_members = defaultdict(list)
        for member in range(self.n):
            group_members[self.find(member)].append(member)
        return group_members

    def __str__(self):
        return '\n'.join(f'{r}: {m}' for r, m in self.all_group_members().items())

#深さ優先探索

point_tree_y=[]
point_tree_x=[]

def dfs(x,y,c2):
	
	#壁
	if y<0 or x<0 or y>=len(detedge) or x>=len(detedge[0]):
		return
	#線じゃない
	if detedge[y][x]<1:
		return
	#見た
	if flag[y*len(detedge[0])+x]==1:
		return
	# detedge[y][x]=c2
	# c2=c2+16
	# c2=c2%255
	point_tree_y.append(y)
	point_tree_x.append(x)
	flag[y*len(detedge[0])+x]=1
	dfs(x+1, y,c2)
	dfs(x-1, y,c2)
	dfs(x, y+1,c2)
	dfs(x, y-1,c2)
	dfs(x+1, y-1,c2)
	dfs(x-1, y-1,c2)
	dfs(x+1, y+1,c2)
	dfs(x-1, y+1,c2)




input = "img/kao.png"

# graying = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
# graying_flip =cv2.flip(graying, 0)
# detedge = cv2.Canny(graying_flip, 100, 1200)



st.title("Drawable Canvas")
st.markdown("""
Draw on the canvas, get the image data back into Python !
* Doubleclick to remove the selected object when not in drawing mode
""")
st.sidebar.header("Configuration")

# Specify brush parameters and drawing mode
b_width = st.sidebar.slider("Brush width: ", 1, 100, 10)
b_color = st.sidebar.color_picker("Enter brush color hex: ")
bg_color = st.sidebar.color_picker("Enter background color hex: ", "#eee")
drawing_mode = st.sidebar.checkbox("Drawing mode ?", True)

# Create a canvas component
canvas_result  = st_canvas(
fill_color="rgba(255, 165, 0, 0.3)",
stroke_width=10,
stroke_color="Black",
background_color="White",
# width = 150,
height= 150,
key="canvas",
)

# Do something interesting with the image data
if canvas_result.image_data is not None:
	st.image(canvas_result.image_data)
	img_gray = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_BGR2GRAY) 
	
	
	detedge = cv2.Canny(img_gray, 100, 1200)
	p=[]


	for i in range(len(detedge)):
		for j in range(len(detedge[0])):
			if detedge[i][j]>0:
				p.append(int(i*len(detedge[0])+j))

	#print(p[3])
	d = {p[i]: i for i in range(len(p))}
	d2 = {i: p[i] for i in range(len(p))}


	flag = {p[i]: 0 for i in range(len(p))}
	ut = UnionFind(len(p))
	for i in p:
		y=int(i/len(detedge[0]))
		x=i%len(detedge[0])
		
		if y-1>=0:
			if x-1>=0:
				if detedge[y-1][x-1]>0:
					ut.union(d[y*len(detedge[0])+x],d[(y-1)*len(detedge[0])+(x-1)])
			if x+1<len(detedge[0]):
				if detedge[y-1][x+1]>0:
					ut.union(d[y*len(detedge[0])+x],d[(y-1)*len(detedge[0])+(x+1)])
			if detedge[y-1][x]>0:
				ut.union(d[y*len(detedge[0])+x],d[(y-1)*len(detedge[0])+(x)])
		if y+1<len(detedge):
			if x-1>=0:
				if detedge[y+1][x-1]>0:
					ut.union(d[y*len(detedge[0])+x],d[(y+1)*len(detedge[0])+(x-1)])
			if x+1<len(detedge[0]):
				if detedge[y+1][x+1]>0:
					ut.union(d[y*len(detedge[0])+x],d[(y+1)*len(detedge[0])+(x+1)])
			if detedge[y+1][x]>0:
				ut.union(d[y*len(detedge[0])+x],d[(y+1)*len(detedge[0])+(x)])
		if x-1>=0:
			if detedge[y][x-1]>0:
				ut.union(d[y*len(detedge[0])+x],d[(y)*len(detedge[0])+(x-1)])
		if x+1<len(detedge[0]):
			if detedge[y][x+1]>0:
				ut.union(d[y*len(detedge[0])+x],d[(y)*len(detedge[0])+(x+1)])
				
	print(ut.roots())

	color = {ut.roots()[i]: (i+1)*40 for i in range(len(ut.roots()))}
	for i in ut.roots():
		dfs(d2[i]%len(detedge[0]),int(d2[i]/len(detedge[0])),100)
	for i in range(len(detedge)):
		print(detedge[i])
		for j in range(len(detedge[0])):
			if detedge[i][j]>0:
				detedge[i][j]=color[ut.find(d[i*len(detedge[0])+j])]
	st.image(detedge)
	print(len(canvas_result.image_data[0]))
	print(len(canvas_result.image_data))
