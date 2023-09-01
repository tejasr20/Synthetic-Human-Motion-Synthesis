from simple_3dviz import Scene
from simple_3dviz import Mesh, Spherecloud
from simple_3dviz.utils import save_frame

scene = Scene(background=(0.0, 0.0, 0.0, 1.0), size=(512, 512))

# Load a mesh from a file
m = Mesh.from_file("/data/tejasr20/summon/predictions/scene/proxd_valid/MPH11_00150_01/human/mesh/human_2.ply")
scene.add(m)

# Create a spherecloud
x = np.linspace(-0.5, 0.5, num=5) 
centers = np.array(np.meshgrid(x, x, x)).reshape(3, -1).T 
colors = np.array([[1, 1, 0, 1], 
                       [0, 1, 1, 1]])[np.random.randint(0, 2, size=centers.shape[0])] 
sizes = np.ones(centers.shape[0])*0.02                                                         
s = Spherecloud(centers, colors, sizes)
scene.add(s)

scene.render()
save_frame("scenes_renderable_1.png", scene.frame)
scene.camera_position = (0, 0.72, -2.3)
scene.render()
save_frame("scenes_renderable_2.png", scene.frame)