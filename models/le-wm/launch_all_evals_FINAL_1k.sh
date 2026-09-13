


# # # RGB horizon5
# python eval.py --config-name=cube_RGB.yaml plan_config.horizon=5
# # RGB horizon10
# python eval.py --config-name=cube_RGB.yaml plan_config.horizon=10

# # # NORMALS horizon5 
# python eval.py --config-name=cube_NRM_OGB.yaml plan_config.horizon=5 
# # NORMALS horizon10
# python eval.py --config-name=cube_NRM_OGB.yaml plan_config.horizon=10 

# # POINTMAPS horizon5
# python eval.py --config-name=cube_POINTMAPS_OGB.yaml plan_config.horizon=5 
# # POINTMAPS horizon10
# python eval.py --config-name=cube_POINTMAPS_OGB.yaml plan_config.horizon=10 

# RGB+D horizon5
python eval.py --config-name=cube_RGB_D_OGB.yaml plan_config.horizon=5
# RGB+D horizon10
python eval.py --config-name=cube_RGB_D_OGB.yaml plan_config.horizon=10

