import os
import trimesh


for idx in range(1, 21):
    print(f"=== Processing usb_cap/{idx} ===")

    # 原始 mesh 所在文件夹：Shape_Data_Mesh/usb_cap/{idx}
    base_dir = os.path.join("Shape_Data_Mesh", "usb_cap", str(idx))

    input_obj1 = os.path.join(base_dir, "partA_new.obj")
    input_obj2 = os.path.join(base_dir, "partB_new.obj")
    vhacd_obj1 = os.path.join(base_dir, "vhacd_partA_new.obj")
    vhacd_obj2 = os.path.join(base_dir, "vhacd_partB_new.obj")

    # ==========================
    #   URDF 目录结构设置
    #   Shape_Data_Mesh/usb_cap/urdf/{idx}/partA.urdf
    # ==========================
    urdf_root = os.path.join("Shape_Data_Mesh", "usb_cap", "urdf")
    urdf_dir = os.path.join(urdf_root, str(idx))   # 比如 .../usb_cap/urdf/1
    os.makedirs(urdf_dir, exist_ok=True)

    urdf_file1 = os.path.join(urdf_dir, "partA.urdf")
    urdf_file2 = os.path.join(urdf_dir, "partB.urdf")

    # 如果 obj 不存在就跳过
    if not os.path.exists(input_obj1) or not os.path.exists(input_obj2):
        print(f"  Skip idx={idx}: obj not found")
        continue
    # Step 1: 加载 mesh
    mesh1 = trimesh.load(input_obj1)
    mesh2 = trimesh.load(input_obj2)
    print(f"  partA faces={len(mesh1.faces)}, verts={len(mesh1.vertices)}")
    print(f"  partB faces={len(mesh2.faces)}, verts={len(mesh2.vertices)}")

    # Step 2: 简化（这里用保留 80% 面数的方式，你也可以改成固定面数）
    target_faces1 = max(100, int(len(mesh1.faces) * 0.8))
    target_faces2 = max(100, int(len(mesh2.faces) * 0.8))

    simple1 = mesh1.simplify_quadric_decimation(target_faces1)
    simple2 = mesh2.simplify_quadric_decimation(target_faces2)

    simple1.export(vhacd_obj1)
    simple2.export(vhacd_obj2)
    print(f"  VHACD (simplified) saved: {vhacd_obj1}")
    print(f"  VHACD (simplified) saved: {vhacd_obj2}")

    # Step 3: 写 URDF
    print(f"  Generating URDF: {urdf_file1}")
    print(f"  Generating URDF: {urdf_file2}")

    urdf_content1 = f"""<?xml version="1.0"?>
<robot name="partA_{idx}">
  <link name="part_link">
    <visual>
      <geometry>
        <mesh filename="{input_obj1}" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{vhacd_obj1}" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
</robot>
"""

    urdf_content2 = f"""<?xml version="1.0"?>
<robot name="partB_{idx}">
  <link name="part_link">
    <visual>
      <geometry>
        <mesh filename="{input_obj2}" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{vhacd_obj2}" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
</robot>
"""

    with open(urdf_file1, "w") as f:
        f.write(urdf_content1)
    with open(urdf_file2, "w") as f:
        f.write(urdf_content2)

    print(f"  URDF written: {urdf_file1}")
    print(f"  URDF written: {urdf_file2}")







# import os
# import trimesh
#
# # input_obj1="Shape_Data_Mesh/usb_cap/1/partA_new.obj"
# # input_obj2="Shape_Data_Mesh/usb_cap/1/partB_new.obj"
# # vhacd_obj1= "Shape_Data_Mesh/usb_cap/1/vhacd_partA_new.obj"
# # vhacd_obj2= "Shape_Data_Mesh/usb_cap/1/vhacd_partB_new.obj"
# # urdf_file1= "Shape_Data_Mesh/usb_cap/1/partA.urdf"
# # urdf_file2= "Shape_Data_Mesh/usb_cap/1/partB.urdf"
#
# # Step 1: Run VHACD
# mesh1= trimesh.load(input_obj1)
# mesh2= trimesh.load(input_obj2)
# print(f"loaded mesh with {len(mesh1.faces)} faces and {len(mesh1.vertices)} vertices")
# print(f"loaded mesh with {len(mesh2.faces)} faces and {len(mesh2.vertices)} vertices")
#
# # simple = mesh.simplify_quadric_decimation(500)
#
# simple1= mesh1.simplify_quadric_decimation(0.8)
# simple2= mesh2.simplify_quadric_decimation(0.8)
# simple1.export(vhacd_obj1)
# simple2.export(vhacd_obj2)
# print(f"VHACD complete. Output saved to: {vhacd_obj1}")
# print(f"VHACD complete. Output saved to: {vhacd_obj2}")
#
# # Step 2: Write URDF
# print(f"Generating URDF: {urdf_file1}")
# print(f"Generating URDF: {urdf_file2}")
# urdf_content1 = f"""<?xml version="1.0"?>
# <robot name="partA">
#   <link name="part_link">
#     <visual>
#       <geometry>
#         <mesh filename="{input_obj1}" scale="1 1 1"/>
#       </geometry>
#     </visual>
#     <collision>
#       <geometry>
#         <mesh filename="{vhacd_obj1}" scale="1 1 1"/>
#       </geometry>
#     </collision>
#     <inertial>
#       <mass value="1"/>
#       <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
#     </inertial>
#   </link>
# </robot>
# """
# urdf_content2 = f"""<?xml version="1.0"?>
# <robot name="partA">
#   <link name="part_link">
#     <visual>
#       <geometry>
#         <mesh filename="{input_obj2}" scale="1 1 1"/>
#       </geometry>
#     </visual>
#     <collision>
#       <geometry>
#         <mesh filename="{vhacd_obj2}" scale="1 1 1"/>
#       </geometry>
#     </collision>
#     <inertial>
#       <mass value="1"/>
#       <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
#     </inertial>
#   </link>
# </robot>
# """
#
# with open(urdf_file1, "w") as f:
#     f.write(urdf_content1)
# with open(urdf_file2, "w") as f:
#     f.write(urdf_content2)
# print(f"URDF file written: {urdf_file1}")
# print(f"URDF file written: {urdf_file2}")
