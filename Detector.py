def car_damage_detector(car_url):
  import detectron2
  from detectron2.utils.logger import setup_logger
  setup_logger()

  # import some common libraries
  import numpy as np
  import os, json, cv2, random
  import matplotlib.pyplot as plt
  import skimage.io as io
  from google.colab.patches import cv2_imshow
  import itertools


  # import some common detectron2 utilities
  from detectron2 import model_zoo
  from detectron2.structures import BoxMode
  from detectron2.engine import DefaultPredictor
  from detectron2.config import get_cfg
  from detectron2.utils.visualizer import Visualizer
  from detectron2.data import MetadataCatalog, DatasetCatalog
  from detectron2.engine import DefaultTrainer
  from detectron2.utils.visualizer import ColorMode
  from detectron2.evaluation import COCOEvaluator, inference_on_dataset
  from detectron2.data import build_detection_test_loader

  # Set base params
  plt.rcParams["figure.figsize"] = [16,9]

  from detectron2.data.datasets import register_coco_instances
  from detectron2.structures import BoxMode

  try:
    for d in ["val_1", "val_2", "val_3", "val_4"]:
        register_coco_instances(f"CarDamage1_{d}", {},
                                f"/content/gdrive/MyDrive/Car damage detection1/Final_model/anno/{d}.json",
                                f"/content/gdrive/MyDrive/Car damage detection1/Final_model/{d}")
  except:
    print("dataset already registered")
  
  #get configuration

  cfg_mul = get_cfg()
  cfg_mul.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  cfg_mul.MODEL.ROI_HEADS.NUM_CLASSES = 6  # only has five classes (headlamp,hood,rear_bumper,front_bumper_door) + 1
  cfg_mul.MODEL.RETINANET.NUM_CLASSES = 6 # only has five classes (headlamp,hood,rear_bumper,front_bumper_door) + 1
  cfg_mul.MODEL.WEIGHTS = os.path.join("/content/gdrive/MyDrive/Car damage detection1/Final_model/part_segmentation_model_v3.pth")
  cfg_mul.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
  cfg_mul['MODEL']['DEVICE']='cpu' #or cpu
  part_predictor = DefaultPredictor(cfg_mul)

  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  
  cfg.MODEL.RETINANET.NUM_CLASSES = 3 
  cfg.MODEL.WEIGHTS = os.path.join("/content/gdrive/MyDrive/Car damage detection1/Final_model/severity_segmentation_model.pth")
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
  cfg['MODEL']['DEVICE']='cpu'#or cuda
  severity_predictor = DefaultPredictor(cfg)

  damage_class_map= {0:'damage'}
  parts_class_map={0:'Windshield',1:'Hood', 2:'Front bumper', 3:'Door', 4:'Back glass', 5:'Rear bumper/Trunk'}
  severity_class_map={0:'minor damage',1:'moderate damage', 2:'severe damage'}


  valdataset_dicts2 = DatasetCatalog.get("CarDamage1_val_4")
  valdataset_dicts3 = DatasetCatalog.get("CarDamage1_val_3")

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize =(30,20))
  im = io.imread(car_url)

  # #part inference
  parts_outputs = part_predictor(im)
  parts_v = Visualizer(im[:, :, ::-1],
                    metadata=MetadataCatalog.get("CarDamage1_val_4"), 
                    scale=0.5, 
                    
  )
  parts_out = parts_v.draw_instance_predictions(parts_outputs["instances"].to("cpu"))

  #severity inference
  severity_outputs = severity_predictor(im)
  severity_v = Visualizer(im[:, :, ::-1],
                      metadata=MetadataCatalog.get("CarDamage1_val_3"), 
                      scale=0.5, 
                      
  )
  severity_out = severity_v.draw_instance_predictions(severity_outputs["instances"].to("cpu"))

  #plot
  ax1.imshow(parts_out.get_image()[:, :, ::-1])
  ax2.imshow(severity_out.get_image()[:, :, ::-1])
  plt.savefig("car_damage.jpeg", bbox_inches='tight',pad_inches = 0)


  parts_prediction_classes = [ parts_class_map[el] + "_" + str(indx) for indx,el in enumerate(parts_outputs["instances"].pred_classes.tolist())]
  severity_prediction_classes = [ severity_class_map[el] + "_" + str(indx) for indx,el in enumerate(severity_outputs["instances"].pred_classes.tolist())]


  def part(dam_class):
    try:
      return (dam_class[0].rsplit('_',1)[0])

    except: 
      return ("Undetectable \n Try taking another clear image of the ")
  def sev(dam_class):
    try:
      return (dam_class[0].rsplit('_',1)[0])

    except: 
      return ("very minor damage / Undetectable damage \n Try taking another clear image")
  parts=part(parts_prediction_classes)
  extents=sev(severity_prediction_classes)
  return parts,extents