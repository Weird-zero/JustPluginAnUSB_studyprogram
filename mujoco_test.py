import mujoco
import mujoco.viewer
import numpy as np
import cv2
import time
def main():
    model=mujoco.MjModel.from_xml_path("mujoco_arm_models/universal_robots_ur5e/scene.xml")
    data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height=400,width=400)
    with mujoco.viewer.launch_passive(model,data)as viewer:
        while viewer.is_running:
            mujoco.mj_step(model,data)
            viewer.sync()
            renderer.update_scene(data, camera="with_camera")
            pixels = renderer.render()
            bgr_image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            cv2.imshow('MuJoCo - Custom Camera View', bgr_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0)
    cv2.destroyAllWindows()
if __name__=="__main__":
    main()
