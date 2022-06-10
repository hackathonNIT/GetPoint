import cv2

from collections import defaultdict


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






input = "img/kao.png"

graying = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
graying_flip =cv2.flip(graying, 0)
detedge = cv2.Canny(graying_flip, 100, 1200)



# cv2.imshow("fas", detedge)


# img_sobel_x = cv2.Sobel(detedge, cv2.CV_32F, 1, 0, ksize=1)
# img_sobel_y = cv2.Sobel(detedge, cv2.CV_32F, 0, 1, ksize=1)
# imgs = cv2.hconcat([img_sobel_x, img_sobel_y])
# cv2.imshow("soblexy",imgs)


ans=0
p=[]


for i in range(len(detedge)):
	for j in range(len(detedge[0])):
		if detedge[i][j]>0:
			ans=ans+1
			p.append(int(i*len(detedge[0])+j))

#print(p[3])
d = {p[i]: i for i in range(len(p))}
d2 = {i: p[i] for i in range(len(p))}


flag = {p[i]: 0 for i in range(len(p))}

#print(d[p[3]])

#print(int(p[3]/len(detedge[0])))
#print(p[3]%len(detedge[0]))

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
# for i in detedge:
# 	print(i)
#ut = UnionFind(len(detedge)*len(detedge[0]))


point_tree_y=[]
point_tree_x=[]



#深さ優先探索
def dfs(x,y,c2):
	
	#壁
	if y<0 or x<0 or y>=len(detedge) or x>=len(detedge[0]):
		return
	#線じゃない
	if detedge[y][x]<10:
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

	
for i in ut.roots():
	dfs(d2[i]%len(detedge[0]),int(d2[i]/len(detedge[0])),100)



	


# for i in range(len(detedge)):
# 	print(detedge[i])
# 	for j in range(len(detedge[0])):
# 		if detedge[i][j]>0:
# 			detedge[i][j]=color[ut.find(d[i*len(detedge[0])+j])]

#ris=cv2.resize(detedge,dsize=(300,300))


xplus= {i:0 for i in range(len(detedge[0]))}

for i in range(len(point_tree_x)):
	point_tree_x[i]=point_tree_x[i]+xplus[int(point_tree_x[i])]
	xplus[int(point_tree_x[i])]=xplus[int(point_tree_x[i])]+0.001




x_sub=[]
y_sub=[]
for i in range(len(point_tree_x)):
	if i%5==0:
		x_sub.append(point_tree_x[i])
		y_sub.append(point_tree_y[i])


print("y1=np.array(",y_sub,")")
#print(len(point_tree_y))

print("x=np.array(",x_sub,")")

cv2.imshow("fas", detedge)
# print(ans)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()