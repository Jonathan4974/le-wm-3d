import numpy as np

# 替换成让你报错的那个具体 npz 文件路径
file_path = "/home/student/users/Public_workspace/le-wm-3DGeom/models/ogbench/visualizations/outputs/multiview_data.npz" 

data = np.load(file_path)
pts3d = data['gt_points']

print("实际类型:", type(pts3d))
print("实际Dtype:", pts3d.dtype)
print("实际形状(Shape):", pts3d.shape)