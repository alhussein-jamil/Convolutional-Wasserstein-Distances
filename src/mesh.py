import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi

#unflattens a 1d array
def i_to_xyz(i,n):
    z = i // (n*n)
    i = i % (n*n)
    y = i // n
    x = i % n
    return (x, y, z)

#flattens a 3d array
def xyz_to_i(x,y,z,n):
    return(n*n*x+n*y+z)


#writes mesh to an .off file
def write_off(filename, verts, faces):
    with open(filename, 'w') as f:
        f.write("OFF\n")
        f.write("%d %d 0\n" % (len(verts), len(faces)))
        for vert in verts:
            f.write("%f %f %f\n" % tuple(vert))
        for face in faces:
            f.write("3 %d %d %d\n" % tuple(face))
            
class Mesh: 
    n = 40
    def __init__(self, n=40):
        self.distribution = None
        self.binary_distribution = None 
        self.binary = None 
        self.count = None 
        self.mesh = None 
        self.points = None
        self.cot_laplacian = None
        self.n = n

    #adds more points to the monte_carlo method
    def update_distribution(self, num_points):
        points=np.random.rand(num_points,3)
        mask = self.mesh.contains(points)
        for p,m in zip(points,mask):
            if(m):
                x = int(p[0]*self.n)
                y = int(p[1]*self.n)
                z = int(p[2]*self.n)
                self.points.append(p)
                self.count[xyz_to_i(x,y,z,self.n)] +=1
        self.distribution= self.count/sum(self.count)
        for i in range(self.n**3):
            if(self.count[i]>0):
                self.binary[i]=1
        self.binary_distribution= self.binary/self.binary.sum()

    #creates a distribution on [0,1]^3 after being cut into n_cuts^3 elements
    def find_distribution(self,n_cuts=20,num_points = 10000):    
        self.n = n_cuts
        points = np.random.rand(num_points, 3)
        self.binary_distribution = np.zeros(n_cuts**3)
        self.binary = np.zeros(n_cuts**3)
        #apply monte carlo method to estimate distribution 
        # Check which points are inside the mesh
        mask = self.mesh.contains(points)
        print("Done masking")
        self.distribution = np.zeros(n_cuts**3)
        inside = []
        for p,m in zip(points,mask):
            if(m):
                x = int(p[0]*n_cuts)
                y = int(p[1]*n_cuts)
                z = int(p[2]*n_cuts)
                inside.append(p)
                self.distribution[xyz_to_i(x,y,z,n_cuts)] +=1
        self.count = self.distribution.copy()
        self.distribution/=sum(self.distribution)
        print("Done Calculating Distribution")
        threshold = max(self.distribution)/2
        for i in range(n_cuts**3):
            if(self.count[i]>0):
                self.binary[i]=1
        self.binary_distribution= self.binary/self.binary.sum()
        self.points = inside 

    #fills empty spaces in our 0-1 cubical mesh
    def fillBin(self):
        binary = self.binary.copy().reshape((self.n,self.n,self.n))
        for i in range(1,self.n-1):
            for j in range(1,self.n-1):
                for k in range(1,self.n-1):
                    if(binary[i,j,k] ==0):
                        if((binary[i,j,k+1]==1 and binary[i,j,k-1]==1) or (binary[i,j+1,k]==1 and binary[i,j-1,k]==1) or (binary[i+1,j,k]==1 and binary[i-1,j,k]==1)  or (binary[i+1,j+1,k+1]==1 and binary[i-1,j-1,k-1]==1)  or (binary[i+1,j+1,k-1]==1 and binary[i-1,j-1,k+1]==1)  or (binary[i+1,j-1,k+1]==1 and binary[i-1,j+1,k-1]==1)  or (binary[i+1,j-1,k-1]==1 and binary[i-1,j+1,k+1]==1)):
                            binary[i,j,k] =1
        self.binary= binary.flatten()
        self.binary_distribution= self.binary/self.binary.sum()

    def normalize_mesh(self):
        # Find the scale factor to fit the bounding box in [0,1]^3
        scale_factor = min(0.90 / (self.mesh.bounds.max(axis=0)-self.mesh.vertices.min(axis=0)))

        # Apply the scale factor to the mesh
        self.mesh.apply_scale(scale_factor)

        # Translate the mesh so that its minimum vertex is at the origin
        self.mesh.apply_translation(-self.mesh.vertices.min(axis=0))
        min_x = self.mesh.vertices[:,0].min()
        min_y = self.mesh.vertices[:,1].min()
        min_z = self.mesh.vertices[:,2].min()
        max_x = self.mesh.vertices[:,0].max()
        max_y = self.mesh.vertices[:,1].max()
        max_z = self.mesh.vertices[:,2].max()
        self.mesh.apply_translation([0.5-(min_x+max_x)/2,0.5-(min_y+max_y)/2,0.5-(min_z+max_z)/2])


    def cotangent_laplacian_with_area(self):
        cot_weights = {}
        for face in self.mesh.faces:
            i, j, k = face
            v1 = self.mesh.vertices[i]
            v2 = self.mesh.vertices[j]
            v3 = self.mesh.vertices[k]
            a = np.linalg.norm(v2 - v3)
            b = np.linalg.norm(v1 - v3)
            c = np.linalg.norm(v1 - v2)
            s = (a+b+c)/2
            #area weight 
            A = np.sqrt(s*(s-a)*(s-b)*(s-c))

            #angles of a triangle
            alpha = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
            beta = np.arccos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c))
            gamma = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))

            #their cot
            cot_alpha = A/np.sin(alpha)
            cot_beta = A/np.sin(beta)
            cot_gamma = A/np.sin(gamma)

            
            cot_weights[(i, j)] = cot_alpha
            cot_weights[(j, k)] = cot_beta
            cot_weights[(k, i)] = cot_gamma
        n = len(self.mesh.vertices)
        laplacian = np.zeros((n, n))
        for (i, j), w in cot_weights.items():
            laplacian[i, i] += w
            laplacian[j, j] += w
            laplacian[i, j] -= w
        self.laplacian = laplacian