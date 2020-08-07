
import numpy as np
import open3d as o3d
from open3d import JVisualizer



# points = (np.random.rand(1000, 3) - 0.5) / 4
# colors = np.random.rand(1000, 3)

def open3d_vis(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    visualizer = JVisualizer()
    visualizer.add_geometry(pcd)
    visualizer.show()
    
# open3d_vis(points, colors)

def open3d_pcd_vis(pcd):
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # if colors is not None:
    #     pcd.colors = o3d.utility.Vector3dVector(colors)

    visualizer = JVisualizer()
    visualizer.add_geometry(pcd)
    visualizer.show()
    
def open3d_init():
    """ # copy and paste this function to the jupyter notebook
    
    """
    import numpy as np
    import open3d as o3d
    from open3d import JVisualizer


    def open3d_vis(points, colors=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        visualizer = JVisualizer()
        visualizer.add_geometry(pcd)
        visualizer.show()

    points = (np.random.rand(1000, 3) - 0.5) / 4
    colors = np.random.rand(1000, 3)

    open3d_vis(points, colors)