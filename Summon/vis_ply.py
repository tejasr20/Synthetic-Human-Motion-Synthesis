from open3d import *    
import tkinter
from pyntcloud import PyntCloud
# import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
import matplotlib
#matplotlib.use('Agg')
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt

def main():
    # cloud = io.read_point_cloud("predictions/scene/proxd_valid/MPH11_00150_01/human/mesh/human_0.ply") # Read point cloud
    # visualization.draw_geometries([cloud])    # Visualize point cloud      
	# img=open3d.io.read_point_cloud("predictions/scene/proxd_valid/MPH11_00150_01/human/mesh/human_0.ply")
	# open3d.io.write_image("0.png",img)
	# human_face = PyntCloud.from_file("predictions/scene/proxd_valid/MPH11_00150_01/human/mesh/human_0.ply")
	# human_face.plot()x
	# plt.plot(human_face)
	
	# plt.plot([1,2,3],[5,7,4])
	# plt.show()
	matplotlib.pyplot.plot([1,2], [3,4], linestyle='-')
	matplotlib.pyplot.show()
	# pylab.savefig('foo.png')
	# plt.savefig("HOOMAN.png")

	# plt.savefig("human.png")

if __name__ == "__main__":
    main()
