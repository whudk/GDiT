import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image
import argparse
import glob
import os
from xml.dom.minidom import parse
import json
from joblib import Parallel, delayed
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
import sys
from torchvision import models, transforms
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import torch

# landcover = {
#   "l1":{
#     "0":"Background",
#   },
#   "l2":{},
#   "l2":{}
# }


landcover = {
  "0000": {
      "name": "Background",
      "subcategories": {},
  },
  "0100": {
    "name": "Cultivated Land",
    "subcategories": {
      "0110": {
        "name": "Paddy Field",
        "subcategories": {}
      },
      "0120": {
        "name": "Dry Land",
        "subcategories": {}
      },
      "0130": {
        "name": "Orchard",
        "subcategories": {
              "0131": {
                "name": "Arbor Fruit Garden",
                "subcategories": {}
              },
              "0132": {
                "name": "Vine Fruit Garden",
                "subcategories": {}
              },
              "0133": {
                "name": "Herbaceous Fruit Garden",
                "subcategories": {}
              }
          },

      },
      "0140": {
          "name": "Tea Garden",
          "subcategories": {}
        },
      "0150": {
        "name": "Mulberry Garden",
        "subcategories": {}
      },
      "0160": {
        "name": "Rubber Plantation",
        "subcategories": {}
      },
      "0170": {
        "name": "Nursery",
        "subcategories": {}
      },
      "0180": {
        "name": "Flower Bed",
        "subcategories": {}
      },
      "0190": {
        "name": "Other Economic Seedlings",
        "subcategories": {
          "0191": {
            "name": "Other Arbor Economic Seedlings",
            "subcategories": {}
          },
          "0192": {
            "name": "Other Vine Economic Seedlings",
            "subcategories": {}
          },
          "0193": {
            "name": "Other Herbaceous Economic Seedlings",
            "subcategories": {}
          }
        }
      },

      }
    },

  "0300": {
    "name": "Forest and Grass Cover",
    "subcategories": {
      "0310": {
        "name": "Arbor Forest",
        "subcategories": {
          "0311": {
            "name": "Broad-leaved Forest",
            "subcategories": {}
          },
          "0312": {
            "name": "Coniferous Forest",
            "subcategories": {}
          },
          "0313": {
            "name": "Mixed Coniferous and Broad-leaved Forest",
            "subcategories": {}
          }
        }
      },
      "0320": {
        "name": "Shrub Forest",
        "subcategories": {
          "0321": {
            "name": "Broad-leaved Shrub Forest",
            "subcategories": {}
          },
          "0322": {
            "name": "Coniferous Shrub Forest",
            "subcategories": {}
          },
          "0323": {
            "name": "Mixed Coniferous and Broad-leaved Shrub Forest",
            "subcategories": {}
          }
        }
      },
      "0330": {
        "name": "Arbor-Shrub Mixed Forest",
        "subcategories": {}
      },
      "0340": {
        "name": "Bamboo Forest",
        "subcategories": {}
      },
      "0360": {
        "name": "Economic Forest",
        "subcategories": {}
      },
      "0370": {
        "name": "Green Land",
        "subcategories": {}
      },
      "0380": {
        "name": "Artificial Young Forest",
        "subcategories": {}
      },
      "0390": {
        "name": "Herbaceous Vegetation",
        "subcategories": {
          "0391": {
            "name": "High Coverage Grassland",
            "subcategories": {}
          },
          "0392": {
            "name": "Medium Coverage Grassland",
            "subcategories": {}
          },
          "0393": {
            "name": "Low Coverage Grassland",
            "subcategories": {}
          }
        }
      },
      "03A0": {
        "name": "Artificial Grassland",
        "subcategories": {
          "03A1": {
            "name": "Pasture",
            "subcategories": {}
          },
          "03A2": {
            "name": "Green Belt",
            "subcategories": {}
          },
          "03A3": {
            "name": "Desert Vegetation",
            "subcategories": {}
          },
          "03A4": {
            "name": "Slope Protection Vegetation",
            "subcategories": {}
          },
          "03A9": {
            "name": "Other Artificial Grassland",
            "subcategories": {}
          }
        }
      }
    }
  },
  "0500": {
    "name": "Buildings",
    "subcategories": {
      "0510": {
        "name": "Multi-storey and Above Residential Buildings",
        "subcategories": {
          "0511": {
            "name": "High Density Multi-storey and Above Residential Buildings",
            "subcategories": {}
          },
          "0512": {
            "name": "Low Density Multi-storey and Above Residential Buildings",
            "subcategories": {}
          }
        }
      },
      "0520": {
        "name": "Low-rise Residential Buildings",
        "subcategories": {
          "0521": {
            "name": "High Density Low-rise Residential Buildings",
            "subcategories": {}
          },
          "0522": {
            "name": "Low Density Low-rise Residential Buildings",
            "subcategories": {}
          }
        }
      },
      "0530": {
        "name": "Mixed Residential Buildings",
        "subcategories": {
        }
      },
      "0540": {
            "name": "Multi-storey Independent Residential Buildings",
            "subcategories": {
              "0541": {
                "name": "Multi-storey Independent Residential Buildings",
                "subcategories": {}
              },
              "0542": {
                "name": "Medium to High-rise Independent Residential Buildings",
                "subcategories": {}
              },
              "0543": {
                "name": "High-rise Independent Residential Buildings",
                "subcategories": {}
              },
              "0544": {
                "name": "Ultra High-rise Independent Residential Buildings",
                "subcategories": {}
                }
            }
        },
      "0550": {
        "name": "Low-rise Independent Residential Buildings",
        "subcategories": {}
        }
      }
    },
  "0600": {
    "name": "Railways and Roads",
    "subcategories": {
      "0610": {
        "name": "Railways",
        "subcategories": {}
      },
      "0620": {
        "name": "Highways",
        "subcategories": {}
      },
      "0630": {
        "name": "Urban Roads",
        "subcategories": {}
      },
      "0640": {
        "name": "Rural Roads",
        "subcategories": {}
      },
      "0650": {
        "name": "Ramp Roads",
        "subcategories": {}
      }
    }
  },
  "0700": {
    "name": "Structures",
    "subcategories": {
      "0710": {
        "name": "Hardened Surface",
        "subcategories": {
          "0711": {
            "name": "Plaza",
            "subcategories": {}
          },
          "0712": {
            "name": "Open-air Sports Fields",
            "subcategories": {}
          },
          "0713": {
            "name": "Open-air Parking Lots",
            "subcategories": {}
          },
          "0714": {
            "name": "Aprons and Runways",
            "subcategories": {}
          },
          "0715": {
            "name": "Hardened Slag",
            "subcategories": {}
          },
          "0716": {
            "name": "Courtyards",
            "subcategories": {}
          },
          "0717": {
            "name": "Open-air Storage Yards",
            "subcategories": {}
          },
          "0718": {
            "name": "Compressed Hardened Surface",
            "subcategories": {}
          },
          "0719": {
            "name": "Other Hardened Surfaces",
            "subcategories": {}
          }
        }
      },
      "0720": {
        "name": "Hydraulic Structures",
        "subcategories": {
          "0721": {
            "name": "Dams",
            "subcategories": {}
          },
          "0722": {
            "name": "Sluices",
            "subcategories": {}
          },
          "0723": {
            "name": "Pumping Stations",
            "subcategories": {}
          },
          "0729": {
            "name": "Other Hydraulic Structures",
            "subcategories": {}
          }
        }
      },
      "0730": {
        "name": "Transport Facilities",
        "subcategories": {
          "0731": {
            "name": "Tunnels",
            "subcategories": {}
          },
          "0732": {
            "name": "Bridges",
            "subcategories": {}
          },
          "0733": {
            "name": "Wharves",
            "subcategories": {}
          },
          "0734": {
            "name": "Ferries",
            "subcategories": {}
          },
          "0735": {
            "name": "Highway Exits and Entrances",
            "subcategories": {}
          },
          "0736": {
            "name": "Gas (Air) and Charging Stations",
            "subcategories": {}
          },
          "0737": {
            "name": "Overpasses",
            "subcategories": {}
          }
        }
      },
       "0740": {
        "name": "City Walls",
        "subcategories": {}
      },
      "0750": {
        "name": "Greenhouses and Sheds",
        "subcategories": {}
      },
      "0760": {
        "name": "Solidification Ponds",
        "subcategories": {
          "0761": {
            "name": "Swimming Pools",
            "subcategories": {}
          },
          "0762": {
            "name": "Sewage Treatment Pools",
            "subcategories": {}
          },
          "0763": {
            "name": "Salt Pans",
            "subcategories": {}
          },
          "0769": {
            "name": "Other Solidification Ponds",
            "subcategories": {}
          }
        }
      },
      "0770": {
        "name": "Industrial Facilities",
        "subcategories": {}
      },
      "0780": {
        "name": "Barracks",
        "subcategories": {}
      },
      "0790": {
        "name": "Other Structures",
        "subcategories": {}
      }
    }
  },
  "0800": {
    "name": "Artificial Excavation Land",
    "subcategories": {
      "0810": {
        "name": "Open-pit Mines",
        "subcategories": {
          "0811": {
            "name": "Open-pit Coal Mines",
            "subcategories": {}
          },
          "0812": {
            "name": "Open-pit Iron Mines",
            "subcategories": {}
          },
          "0813": {
            "name": "Open-pit Copper Mines",
            "subcategories": {}
          },
          "0814": {
            "name": "Open-pit Quarries",
            "subcategories": {}
          },
          "0815": {
            "name": "Open-pit Clay Mines",
            "subcategories": {}
          },
          "0819": {
            "name": "Other Open-pit Mines",
            "subcategories": {}
          }
        }
      },
      "0820": {
        "name": "Dumping Grounds",
        "subcategories": {
          "0821": {
            "name": "Tailings",
            "subcategories": {}
          },
          "0822": {
            "name": "Landfills",
            "subcategories": {}
          },
          "0829": {
            "name": "Other Dumping Grounds",
            "subcategories": {}
          }
        }
      },
      "0830": {
        "name": "Construction Sites",
        "subcategories": {
          "0831": {
            "name": "Demolition Sites",
            "subcategories": {}
          },
          "0832": {
            "name": "Building Construction Sites",
            "subcategories": {}
          },
          "0833": {
            "name": "Road Construction Sites",
            "subcategories": {}
          },
          "0839": {
            "name": "Other Construction Sites",
            "subcategories": {}
          }
        }
      },
      "0890": {
        "name": "Other Artificial Excavation Land",
        "subcategories": {}
      }
    }
  },
  "0900": {
    "name": "Desert and Bare Land",
    "subcategories": {
      "0910": {
        "name": "Saline Land",
        "subcategories": {}
      },
      "0920": {
        "name": "Mud Land",
        "subcategories": {}
      },
      "0930": {
        "name": "Sandy Land",
        "subcategories": {}
      },
      "0940": {
        "name": "Gravel Land",
        "subcategories": {}
      },
      "0950": {
        "name": "Rocky Land",
        "subcategories": {}
      }
    }
  },
  "1000": {
    "name": "Water Area",
    "subcategories": {
      "1010": {
        "name": "River Channel",
        "subcategories": {
          "1011": {
            "name": "River Flow",
            "subcategories": {}
          },
          "1012": {
            "name": "River Bed",
            "subcategories": {}
          }
        }
      },
      "1020": {
        "name": "Lake",
        "subcategories": {}
      },
      "1030": {
        "name": "Reservoir",
        "subcategories": {
          "1031": {
            "name": "Reservoir Water Body",
            "subcategories": {}
          },
          "1032": {
            "name": "Reservoir Bank",
            "subcategories": {}
          }
        }
      },
      "1040": {
        "name": "Sea Surface",
        "subcategories": {}
      },
      "1050": {
        "name": "Glaciers and Permanent Snow",
        "subcategories": {
          "1051": {
            "name": "Glacier",
            "subcategories": {}
          },
          "1052": {
            "name": "Permanent Snow",
            "subcategories": {}
          }
        }
      }
    }
  },
  "1100": {
    "name": "Geographic Units",
    "subcategories": {
      "1110": {
        "name": "Administrative Planning and Management Units",
        "subcategories": {
          "1111": {
            "name": "National Administrative Regions",
            "subcategories": {}
          },
          "1112": {
            "name": "Provincial Administrative Regions",
            "subcategories": {}
          },
          "1113": {
            "name": "Special Administrative Regions",
            "subcategories": {}
          },
          "1114": {
            "name": "Municipal and Prefectural Administrative Regions",
            "subcategories": {}
          },
          "1115": {
            "name": "County Administrative Regions",
            "subcategories": {}
          },
          "1116": {
            "name": "Township Administrative Regions",
            "subcategories": {}
          },
          "1117": {
            "name": "Administrative Villages",
            "subcategories": {}
          },
          "1118": {
            "name": "Urban Centers",
            "subcategories": {}
          },
          "1119": {
            "name": "Other Special Administrative Management Units",
            "subcategories": {}
          }
        }
      },
      "1120": {
        "name": "Socio-economic Regional Units",
        "subcategories": {
          "1121": {
            "name": "Main Functional Zones",
            "subcategories": {}
          },
          "1122": {
            "name": "Development and Tax-free Zones",
            "subcategories": {}
          },
          "1123": {
            "name": "State-owned Farms, Forests, and Pastures",
            "subcategories": {}
          },
          "1124": {
            "name": "Natural and Cultural Protection Zones",
            "subcategories": {}
          },
          "1125": {
            "name": "Natural and Cultural Heritage Sites",
            "subcategories": {}
          },
          "1126": {
            "name": "Scenic and Tourist Areas",
            "subcategories": {}
          },
          "1127": {
            "name": "Forest Parks",
            "subcategories": {}
          },
          "1128": {
            "name": "Geological Parks",
            "subcategories": {}
          },
          "1129": {
            "name": "Xingang and Peixiang Flood Control Areas",
            "subcategories": {}
          },
          "112A": {
            "name": "Drinking Water Source Protection Zones",
            "subcategories": {}
          },
          "112B": {
            "name": "Ecological Red Line Zones",
            "subcategories": {}
          },
          "112C": {
            "name": "Permanent Basic Farmland Protection Zones",
            "subcategories": {}
          },
          "112D": {
            "name": "Urban Development Boundaries",
            "subcategories": {}
          }
        }
      },
      "1130": {
        "name": "Natural Geographic Units",
        "subcategories": {
          "1131": {
            "name": "Watersheds",
            "subcategories": {}
          },
          "1132": {
            "name": "Landform Units",
            "subcategories": {}
          },
          "1133": {
            "name": "Geological Type Units",
            "subcategories": {}
          },
          "1134": {
            "name": "Wetland Protection Zones",
            "subcategories": {}
          },
          "1135": {
            "name": "Swamps",
            "subcategories": {}
          }
        }
      },
      "1140": {
        "name": "Urban Comprehensive Functional Units",
        "subcategories": {
          "1141": {
            "name": "Residential Areas",
            "subcategories": {}
          },
          "1142": {
            "name": "Industrial Enterprises",
            "subcategories": {}
          },
          "1143": {
            "name": "Institutional Units",
            "subcategories": {}
          },
          "1144": {
            "name": "Recreation and Scenic Areas",
            "subcategories": {}
          },
          "1145": {
            "name": "Sports Facilities",
            "subcategories": {}
          },
          "1146": {
            "name": "Famous Sites",
            "subcategories": {}
          },
          "1147": {
            "name": "Religious Places",
            "subcategories": {}
          },
          "1148": {
            "name": "Protected Residential Buildings",
            "subcategories": {}
          },
          "1149": {
            "name": "Parking Areas",
            "subcategories": {}
          }
        }
      }
    }
  }
}


oem = {
  "l1":{
    "0":"Background",
      "1":"Bareland",
      "2":"Rangeland",
      "3":"Developed space",
      "4":"Road",
      "5":"Tree",
      "6":"Water",
      "7":"Agriculture land",
      "8":"Building"
  },
    "l2":{},
    "l3":{}
}

def get_mapping_value(xml):
  transBM = parse(xml)
  root = transBM.documentElement

  mapping_value = {}
  all_codes = root.getElementsByTagName('BM')
  for bm in all_codes:
      key =  bm.attributes['key'].value
      val = int( bm.attributes['val'].value)
      mapping_value[val] = key

  return mapping_value


def get_caps_mapping_val(landcover, xml):
  transBM = parse(xml)
  root = transBM.documentElement

  caps_mapping_value = {}
  all_codes = root.getElementsByTagName('BM')
  for bm in all_codes:
    key = bm.attributes['key'].value
    name = find_landcover_name_with_code(landcover, key)
    val = int(bm.attributes['val'].value)
    caps_mapping_value[val] = name

  return caps_mapping_value




# {
#   "caption":这是一个 [landcover / landuse] 分类图，其中一级类为[L1_1,L1_2,L1_3].二级类为[L2_1, L2_2, L3_3].三级类为[L3_1, L3_2, L3_3]

#   'Level-1': {
#     '0': {'role': 'L1',  'words': ['L1_1']},
#     '1': {'role': 'L1',  'words': ['L1_2']},
#     '2': {'role': 'L1',  'words': ['L1_3']},
#     'adj':[[0,1], [1, 2]]
#   },
#   'Level-2': {
#     '0': {'role': 'L2',  'words': ['L2_1']},
#     '1': {'role': 'L2',  'words': ['L2_2']},
#     '2': {'role': 'L2',  'words': ['L2_3']},
#     'adj':[[0,1], [1, 2]]
#   },
#   'Level-3': {
#     '0': {'role': 'L3',  'words': ['L2_1']},
#     '1': {'role': 'L3',  'words': ['L2_2']},
#     '2': {'role': 'L3',  'words': ['L2_3']},
#     'adj':[[0,1], [1, 2]]
#   },
 
#   'contains_12': [[0, 1], [1, 1], [1, 2]]
#   'contains_23': [[0, 1], [1, 1], [1, 2]]
# }

def find_parent_class(landcover, code):
  def find_parent_recursive(current_level, current_code):
    for subcode, subclass in current_level.items():
      if current_code in subclass["subcategories"]:
        return subclass["name"]
      parent = find_parent_recursive(subclass["subcategories"], current_code)
      if parent:
        return parent
    return None

  parent_class = find_parent_recursive(landcover, code)
  return parent_class if parent_class else "Code not found"


def find_landcover_name_with_code(landcover, code):
  def find_class_recursive(current_level, current_code):
    for subcode, subclass in current_level.items():
      if subcode == current_code:
        return subclass["name"]
      result = find_class_recursive(subclass["subcategories"], current_code)
      if result:
        return result
    return None

  class_name = find_class_recursive(landcover, code)
  return class_name if class_name else "Unknown"

def find_landcover_class(landcover, code):
  def find_class_recursive(current_level, current_code, hierarchy):
    for subcode, subclass in current_level.items():
      if subcode == current_code:
        return hierarchy + [subclass["name"]]
      result = find_class_recursive(subclass["subcategories"], current_code, hierarchy + [subclass["name"]])
      if result:
        return result
    return None

  hierarchy = find_class_recursive(landcover, code, [])
  if not hierarchy:
    return "Code not found"

  if len(hierarchy) == 1:
    return {"l1": hierarchy[0]}
  elif len(hierarchy) == 2:
    return {"l1": hierarchy[0], "l2": hierarchy[1]}
  elif len(hierarchy) == 3:
    return {"l1": hierarchy[0], "l2": hierarchy[1], "l3": hierarchy[2]}
  else:
    return "Invalid hierarchy length"

def categorize_positions(seg_map, mapping_value,landcover):
    category_positions = {}
    for i in range(seg_map.shape[0]):
      for j in range(seg_map.shape[1]):
        val = seg_map[i, j]
        if val not in category_positions:
          category_positions[val] = {
            "l1": [],
            "l2": [],
            "l3": []
          }
        classification = find_landcover_class(landcover, mapping_value[val])

        if "l1" in classification:
        #if classification["l1"]:
          category_positions[val]["l1"].append((i, j))
        if "l2" in classification:
          category_positions[val]["l2"].append((i, j))
        if "l3" in classification:
          category_positions[val]["l3"].append((i, j))

    return category_positions


#

def generate_dict(landcover):
  code_name_dict = {}

  def traverse_landcover(current_level):
    for code, info in current_level.items():
      code_name_dict[code] = info["name"]
      traverse_landcover(info["subcategories"])

  traverse_landcover(landcover)


  # 使用字典推导式互换键和值
  code_name_dict = {v: k for k, v in code_name_dict.items()}
  return code_name_dict
def find_landcover_class_with_code(landcover, code):
  def find_class_recursive(current_level, current_code, hierarchy):
    for subcode, subclass in current_level.items():
      if subcode == current_code:
        return hierarchy + [subclass["name"]]
      result = find_class_recursive(subclass["subcategories"], current_code, hierarchy + [subclass["name"]])
      if result:
        return result
    return None

  hierarchy = find_class_recursive(landcover, code, [])
  if not hierarchy:
    return "Code not found"

  if len(hierarchy) == 1:
    return {"L1": hierarchy[0], "L2": None, "L3": None}
  elif len(hierarchy) == 2:
    return {"L1": hierarchy[0], "L2": hierarchy[1], "L3": None}
  elif len(hierarchy) == 3:
    return {"L1": hierarchy[0], "L2": hierarchy[1], "L3":  hierarchy[2]}
  else:
    return "Invalid hierarchy length"


def find_parent_code(landcover, target_code):
  def find_class_recursive(current_level, current_code, parent_code=None):
    for subcode, subclass in current_level.items():
      if subcode == current_code:
        return parent_code
      result = find_class_recursive(subclass["subcategories"], current_code, subcode)
      if result:
        return result
    return None

  parent_code = find_class_recursive(landcover, target_code)
  return parent_code if parent_code else "Parent code not found"
def update_segMap(seg_map, mapping_value,  name_bm_dict, bm_num_dict,   landcover, stage = 1):
    value_list = np.unique(seg_map)
    reversed_mapping_value = {v: k for k, v in mapping_value.items()}
    new_seg = seg_map.copy()
    if stage == 1:
        for value in value_list:
           #判断当前是属于哪个类，如果是L1,  continue, 如果为L2, L3,返回一级类的编码，并重新映射seg_map
            code = mapping_value[value]
            classification = find_landcover_class_with_code(landcover, str(code))
            if classification == "Code not found":
                continue
            if classification["L3"] is not None:
                parent_code = find_parent_code(landcover, name_bm_dict[classification["L3"]])
                if parent_code:
                      grandparent_code = find_parent_code(landcover, parent_code)
                      if grandparent_code:
                          new_seg[seg_map == value] = reversed_mapping_value[grandparent_code]
            elif classification["L2"] is not None:
                parent_code = find_parent_code(landcover, name_bm_dict[classification["L2"]])
                if parent_code:
                   new_seg[seg_map == value] = reversed_mapping_value[parent_code]
    elif stage == 2:
        for value in value_list:
          code = mapping_value[value]
          classification = find_landcover_class_with_code(landcover, str(code))
          if classification == "Code not found":
            continue
          elif classification["L3"] is not None:
            parent_code = find_parent_code(landcover, name_bm_dict[classification["L3"]])
            if parent_code:
              new_seg[seg_map == value] = reversed_mapping_value[parent_code]
    return new_seg
def get_semantic_map_captions(seg_path, mapping_value, categories):
    # 读取语义图
    seg_map = np.array(Image.open(seg_path))
    #从 seg_map 向上合并为二级类

    seg_map_stage_1 = update_segMap(seg_map, mapping_value,landcover)
    #seg_map_second =

    #从二级类向上合并为一级类


    # 初始化分类数据
    level_1 = {}
    level_2 = {}
    level_3 = {}

    adj_level_1 = []
    adj_level_2 = []
    adj_level_3 = []

    contains_12 = []
    contains_23 = []

    # 用于记录类别出现的位置
    category_positions = {}
    category_positions = categorize_positions(seg_map, mapping_value, landcover)
    # 遍历语义图，记录每个类别的位置
    # for i in range(seg_map.shape[0]):
    #     for j in range(seg_map.shape[1]):
    #         val = seg_map[i, j]
    #         if val not in category_positions:
    #             category_positions[val] = []
    #         category_positions[val].append((i, j))

    # 根据位置确定相邻关系
    id0, id1, id2 = 0, 0 ,0
    for val, positions in category_positions.items():
        if val in mapping_value:
            key = mapping_value[val]
            category = find_landcover_class( landcover, key)  #获取编码的一级类，二级类，三级类，如果是三级类，则找到他的二级类，一级类， 如果是二级类则找到他的一级类，一级类直接返回

            if category:
                for l in category.keys():
                    if l == "l1":
                      level_1[str(id0)] = {"role": "L1", "words": [category[l]]}
                      id0 = id0 + 1
                    elif l == "l2":
                      level_2[str(id1)] = {"role": "L2", "words": [category[l]]}
                      # parent = find_parent_class(landcover, [category[l]])
                      contains_12.append([id0 - 1, id1])
                      id1 = id1 + 1
                    elif l == "l3":
                      level_3[str(id2)] = {"role": "L3", "words": [category[l]]}
                      contains_12.append([id1 - 1, id2])
                      id2 = id2 + 1

                #level = len(category) - 1
                # if level == 0:
                #     level_1[key] = {"role": "L1", "words": [category["l1"]]}
                # elif level == 1:
                #     level_2[key] = {"role": "L2", "words": [category["l2"]]}
                #     parent_key = key[:2]
                #     contains_12.append([list(level_1.keys()).index(parent_key), list(level_2.keys()).index(key)])
                # elif level == 2:
                #     level_3[key] = {"role": "L3", "words": [category["l3"]]}
                #     parent_key = key[:4]
                #     contains_23.append([list(level_2.keys()).index(parent_key), list(level_3.keys()).index(key)])

    # 确定每个层级内部的相邻关系
    def get_adj_relations(level_data):
        adj = []
        keys = list(level_data.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if are_adjacent(category_positions[mapping_value[keys[i]]], category_positions[mapping_value[keys[j]]]):
                    adj.append([i, j])
        return adj

    adj_level_1 = get_adj_relations(level_1)
    adj_level_2 = get_adj_relations(level_2)
    adj_level_3 = get_adj_relations(level_3)

    # 生成最终的 JSON 数据
    result = {
        "caption": "这是一个 landcover 分类图，其中一级类为[L1_1, L1_2, L1_3]. 二级类为[L2_1, L2_2, L2_3]. 三级类为[L3_1, L3_2, L3_3]",
        "Level-1": {str(i): data for i, data in enumerate(level_1.values())},
        "Level-2": {str(i): data for i, data in enumerate(level_2.values())},
        "Level-3": {str(i): data for i, data in enumerate(level_3.values())},
        "Level-1": {"adj": adj_level_1},
        "Level-2": {"adj": adj_level_2},
        "Level-3": {"adj": adj_level_3},
        "contains_12": contains_12,
        "contains_23": contains_23
    }

    return result

def are_adjacent(pos_list_1, pos_list_2):
    for pos1 in pos_list_1:
        for pos2 in pos_list_2:
            if abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1:
                return True
    return False


def visualize_segmentation_maps(*maps, titles):
    num_maps = len(maps)
    plt.figure(figsize=(15, 5))
    for i, seg_map in enumerate(maps, start=1):
        plt.subplot(1, num_maps, i)
        plt.imshow(seg_map, cmap='tab20')  # 使用'tab20'色彩映射，可以根据需要更改
        plt.title(titles[i - 1])
        plt.axis('off')
    plt.show()


def calculate_inclusion_relationship(seg_map_stage_1, seg_map_stage_2):
  relationship_dict = {}

  # 获取所有唯一的一级类
  unique_classes_stage_1 = np.unique(seg_map_stage_1)
  unique_classes_stage_2 = np.unique(seg_map_stage_2)
  for i, class_1 in enumerate(unique_classes_stage_1):
    # 找到对应一级类在stage_1中的位置
    mask_stage_1 = seg_map_stage_1 == class_1

    # 使用mask在stage_2中找到对应的二级类
    corresponding_classes_stage_2 = np.unique(seg_map_stage_2[mask_stage_1])

    # 将对应的二级类添加到字典中

    corresponding_classes_indices = [np.where(unique_classes_stage_2 == class_2)[0][0] for class_2 in
                                     corresponding_classes_stage_2]
    relationship_dict[
      i] = corresponding_classes_indices  # 返回编号corresponding_classes_stage_2 在  unique_classes_stage_2的编号
  return relationship_dict
def calculate_adjacency_list_8_neighborhood(seg_map):
  adjacency_set = set()
  height, width = seg_map.shape

  for i in range(height):
    for j in range(width):
      current_class = seg_map[i, j]
      # 定义8邻域方向
      directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
      for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < height and 0 <= nj < width:
          neighbor_class = seg_map[ni, nj]
          if current_class != neighbor_class:
            adjacency_set.add(tuple(sorted((current_class, neighbor_class))))

  # 将集合转换为列表
  adjacency_list = [list(pair) for pair in adjacency_set]
  return adjacency_list


import scipy.ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import dilation, square

#from skimage.measure import label
from scipy.ndimage import label
from scipy.ndimage import binary_dilation


def get_boxes_and_captions(semantic_map, mapping_value):
  instance_map = np.zeros_like(semantic_map, dtype=np.int32)
  instance_id = 1
  boxes_info = []

  # Step 1: 实例划分（按主类 label 分开）
  unique_classes = np.unique(semantic_map)
  for semantic_class in unique_classes:
    if semantic_class == 0:
      continue
    binary_mask = (semantic_map == semantic_class).astype(np.uint8)
    labeled_components, num_labels = label(binary_mask)
    props = regionprops(labeled_components)

    for prop in props:
      if prop.area < 20:
        continue
      region_mask = (labeled_components == prop.label)
      instance_map[region_mask] = instance_id

      # 获取旋转框
      mask = region_mask.astype(np.uint8)
      contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      if not contours:
        continue
      rect = cv2.minAreaRect(contours[0])
      box = cv2.boxPoints(rect)
      box = np.int0(box).tolist()

      # 在框内找所有语义标签
      h, w = semantic_map.shape
      box_mask = np.zeros((h, w), dtype=np.uint8)
      cv2.drawContours(box_mask, [np.array(box)], -1, 1, -1)  # 填充旋转框区域
      region_inside_box = semantic_map[box_mask == 1]

      # 负类（不同于主类的）
      included_ids = np.unique(region_inside_box)
      included_ids = included_ids[(included_ids != 0) & (included_ids != semantic_class)]
      included_classes = [mapping_value[cid] for cid in included_ids]

      # 构造语义描述
      main_class = mapping_value[semantic_class]
      # if included_classes:
      #   included_str = ", ".join(included_classes)
      #   caption = f"This node primarily belongs to category {main_class}, which also includes [{included_str}]."
      # else:
      caption = f"This node primarily belongs to category {main_class}."

      boxes_info.append({
        "instance_id": instance_id,
        "box": box,
        "main_class": main_class,
        "included_classes": included_classes,
        "caption": caption
      })

      instance_id += 1

  return instance_map, boxes_info


def semantic_to_instance_map(semantic_map, mapping_value):
  """
  Converts a semantic map to an instance map.
  Each connected region within a semantic class is assigned a unique instance ID.

  Parameters:
      semantic_map (np.ndarray): 2D array where each unique value represents a semantic class.

  Returns:
      np.ndarray: Instance map of the same shape as the input, where each connected component
                  of a semantic class has a unique instance ID.
  """
  instance_map = np.zeros_like(semantic_map, dtype=np.int32)
  instance_id = 1  # Start instance IDs from 1

  unique_classes = np.unique(semantic_map)  # Get unique semantic labels

  captions = []
  for semantic_class in unique_classes:
    # Create a binary mask for the current semantic class
    if semantic_class == 0:
      continue
    binary_mask = (semantic_map == semantic_class)
    labeled_components, num_components = label(binary_mask)

    # 获取所有连通区域属性（如面积）
    props = regionprops(labeled_components)

    for prop in props:
      if prop.area < 20:
        continue  # 跳过小区域

      # 当前区域的掩码
      region_mask = (labeled_components == prop.label)

      # 分配 instance_id
      instance_map[region_mask] = instance_id
      captions.append(f"{mapping_value[semantic_class]}")
      instance_id += 1

  return instance_map, captions

def get_adjacency_matrix_from_instace_map(instance_map):
      """
      Computes the adjacency matrix for objects in an instance map.
      Two objects are considered adjacent if they share a boundary.

      Parameters:
          instance_map (np.ndarray): 2D array where each unique value represents a different object.

      Returns:
          np.ndarray: Adjacency matrix of shape (num_objects, num_objects).
      """
      num_objects = int(instance_map.max())  # Assuming object labels start at 1
      adjacency_matrix = np.zeros((num_objects, num_objects), dtype=int)  # Include background if necessary

      # Define the neighborhood (4-connectivity)
      structuring_element = np.array([[0, 1, 0],
                                       [1, 1, 1],
                                       [0, 1, 0]])

      # Iterate over each object in the instance map
      for obj_label in range(1, num_objects + 1):  # Assuming 0 is background
          # Create a binary mask for the current object
          binary_mask = (instance_map == obj_label)

          # Expand the current object's boundary using dilation
          dilated_mask = binary_dilation(binary_mask, structure=structuring_element)

          # Identify neighboring objects by overlapping dilated regions
          neighbors = np.unique(instance_map[dilated_mask])
          neighbors = neighbors[(neighbors != obj_label) & (neighbors != 0)]  # Exclude self and background

          # Update adjacency matrix
          for neighbor in neighbors:
              adjacency_matrix[obj_label -1, neighbor -1] = 1
              adjacency_matrix[neighbor -1 , obj_label -1] = 1  # Symmetry for undirected graph

      # Return the adjacency matrix without the 0-label row/column (if background is ignored)
      #adjacency_matrix = adjacency_matrix[1:, 1:] if 0 in instance_map else adjacency_matrix

      # Find indices where the adjacency matrix has 1 (indicating an edge)
      rows, cols = np.where(adjacency_matrix > 0)

      # Stack them to create edge indices
      edge_index = np.stack([rows, cols], axis=0)

      return edge_index


def get_adjacency_matrix(semantic_map,mapping_value):
  labeled_map = np.zeros_like(semantic_map, dtype=int)
  current_label = 1  # 用来给每个对象分配唯一标签
  captions = []  # 用来存储每个对象的caption

  # 针对每个类别，找到连通区域并标记
  for value, caption in mapping_value.items():
    # 对于每个类别，找到连通区域
    mask = (semantic_map == int(value))
    # Label connected components using 8-connectivity
    labeled_submap = label(mask, connectivity=2)
    # 过滤小区域
    regions = regionprops(labeled_submap)
    for region in regions:
      if region.area < 20:  # 面积阈值
        labeled_submap[labeled_submap == region.label] = 0

    # 重新标记
    labeled_submap = label(labeled_submap > 0, connectivity=2)
    num_features = labeled_submap.max()  # 获取连通区域数量


    # 更新整体的labeled_map，并为每个对象生成caption
    for i in range(1, num_features + 1):
      labeled_map[labeled_submap == i] = current_label
      captions.append(f"{caption}")
      current_label += 1

  # 计算邻接矩阵
  num_objects = current_label - 1
  adjacency_matrix = np.zeros((num_objects, num_objects), dtype=int)

  # 为了计算邻接，扩展每个对象的边界，并检测是否有重叠
  for i in range(1, num_objects + 1):
    # 取出第i个对象的掩码
    object_mask_i = (labeled_map == i)

    # 对对象边界进行膨胀操作，扩大边界区域
    dilated_object_i = dilation(object_mask_i, square(3))  # 膨胀半径可以调整

    for j in range(i + 1, num_objects + 1):
      # 取出第j个对象的掩码
      object_mask_j = (labeled_map == j)

      # 检查两个对象的边界是否相邻（是否有重叠）
      if np.any(dilated_object_i & object_mask_j):
        adjacency_matrix[i - 1, j - 1] = 1
        adjacency_matrix[j - 1, i - 1] = 1  # 对称关系

      # Visualization (optional)
  # plt.figure(figsize=(10, 5))
  #
  # # Display labeled map
  # plt.subplot(1, 2, 1)
  # plt.imshow(labeled_map, cmap='nipy_spectral')
  # plt.title("Labeled Map")
  #
  # props = regionprops(labeled_map)
  # for prop in props:
  #   # Display label at the centroid of each object
  #   y, x = prop.centroid
  #   plt.text(x, y, str(int(prop.label)), color='white', fontsize=12, ha='center', va='center')
  #
  # # Display semantic map
  # plt.subplot(1, 2, 2)
  # plt.imshow(semantic_map, cmap='gray', vmin=0, vmax=10)
  # plt.title("Semantic Map")
  #
  # plt.tight_layout()
  # plt.show()

  return adjacency_matrix, captions, labeled_map
  #return adjacency_matrix, captions, labeled_map

def get_adj_relations(level_data, adj_list, bm_num_dict ):

  if len(adj_list) == 0:
     return []

  name_num_dict = {v: k for k, v in bm_num_dict.items()}

  def replace_adj_list_values(adj_list, target_value, replacement_value):
      return [[replacement_value if val == target_value else val for val in sublist] for sublist in adj_list]
  for i, smp in enumerate(range(len(level_data))):
      words = level_data[str(smp)]["words"]
      words_to_val = int(name_num_dict[words])
      update_adj_list = replace_adj_list_values(adj_list,  words_to_val , i)
      adj_list = update_adj_list
  return  update_adj_list

def get_node_feats(model, captions):
  word_embeddings = dict()
  for key, val in captions.items():
    with torch.no_grad():
      word_embeddings[val] = model(["This node is " +  val  + "."]).cpu()
  return word_embeddings

def show(x_cond_np, instance_map_np, edges):
    # Assuming x_cond is [Batch, Height, Width] and instance_map is [Batch, Height, Width] with graph data


    # Visualize the first instance from each
    plt.figure(figsize=(15, 5))

    # Plot x_cond
    plt.subplot(1, 3, 1)
    plt.imshow(x_cond_np,
               cmap="viridis")  # Display the first sample with a color map suitable for range 0-9
    plt.title("x_cond")
    plt.colorbar(label="Value Range (0-9)")
    plt.axis("off")

    # Plot instance_map
    plt.subplot(1, 3, 2)
    plt.imshow(instance_map_np, cmap="tab20")  # Assuming instance_map is categorical
    plt.title("instance_map")
    plt.colorbar(label="Instance IDs")
    plt.axis("off")

    # Plot graph on instance_map
    plt.subplot(1, 3, 3)
    plt.imshow(instance_map_np, cmap="tab20")  # Use the instance map as background
    plt.title("Graph on Instance Map")
    plt.axis("off")

    # Extract nodes from graph and overlay them
    # Extract nodes and edges from graph and overlay them
    edges = edges.T # Assuming edges is a tensor[2,edges]

    # Plot edges using instance_map for positioning
    # for start_idx, end_idx in edges:
    #     start_instance = np.argwhere(instance_map_np[0] == start_idx).float().mean(axis = 1)
    #     end_instance = np.argwhere(instance_map_np[0] == end_idx).float().mean(axis = 1)
    #     plt.plot([start_instance[1], end_instance[0]], [start_instance[0], end_instance[0]],
    #              color="blue", linewidth=1, alpha=0.7)

    # Plot nodes and edges

    node_positions = []  # List to store positions of nodes
    for idx in range(instance_map_np.max()):
      # Calculate node position based on the instance map
      node_instance = np.argwhere(instance_map_np ==  (idx+1)).mean(axis=0)
      node_positions.append(node_instance)
      plt.scatter(node_instance[1], node_instance[0], color="red", s=50,
                  label=f"Node {idx}" if idx == 0 else "")

    # Plot edges based on adjacency using node positions
    for start_idx, end_idx in edges:
      # Retrieve start and end node positions
      start_position = node_positions[start_idx.item()]
      end_position = node_positions[end_idx.item()]

      # Draw a line (edge) between the start and end positions
      plt.plot([start_position[1], end_position[1]],  # X-coordinates
               [start_position[0], end_position[0]],  # Y-coordinates
               color="blue", linewidth=1, alpha=0.7)

      # plot edges if nodes are adj

    plt.legend(loc="upper right")

    plt.show()




def convert_oem_to_captions(node_dict, segmap, mapping_value, out_dir):

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    feats_dir = os.path.join(out_dir, 'node_feats')
    os.makedirs(feats_dir, exist_ok=True)
    edges_dir = os.path.join(out_dir, 'edge_indexs')
    os.makedirs(edges_dir, exist_ok=True)
    instances_dir = os.path.join(out_dir, 'instancemap')
    os.makedirs(instances_dir, exist_ok=True)


    seg_map = cv2.imread(segmap,cv2.IMREAD_GRAYSCALE)
    instance_map, node_captions = semantic_to_instance_map(seg_map, mapping_value)


    edges = get_adjacency_matrix_from_instace_map(instance_map)

    #adj, captions, instance_map = get_adjacency_matrix(seg_map, mapping_value)

    basename = os.path.basename(os.path.basename(segmap))
    np.save(os.path.join(instances_dir, basename.replace(".tif", "_inst.npy")),instance_map)

    node_l1 = []
    for caption in node_captions:
      node_l1.append(node_dict[caption])
    node_l1 = torch.cat((node_l1),dim = 0)
    #node_l1 = get_node_feats(model, captions)
    l1_edge = edges


    #show(seg_map,instance_map,l1_edge)

    np.save(os.path.join(feats_dir, basename.replace(".tif", "_l1.npy")), node_l1.numpy())
    np.save(os.path.join(edges_dir, basename.replace(".tif", "_l1_edge.npy")), l1_edge)
    # if node_l2 is not None and l2_edge is not None:
    #   np.save(os.path.join(feats_dir, basename.replace(".json", "_l2.npy")), node_l2.numpy())
    #   np.save(os.path.join(edges_dir, basename.replace(".json", "_l2_edge.npy")), l2_edge.numpy())
    # if node_l3 is not None and l3_edge is not None:
    #   np.save(os.path.join(feats_dir, basename.replace(".json", "_l3.npy")), node_l3.numpy())
    #   np.save(os.path.join(edges_dir, basename.replace(".json", "_l3_edge.npy")), l3_edge.numpy())

    pass
    return
    adj= calculate_adjacency_list_8_neighborhood(seg_map)


    level_1 = {val: mapping_value[str(val)] for val in np.unique(seg_map)}
    level_1 = {str(i): {'role': 'L1', 'words': data} for i, data in enumerate(level_1.values())}

    adj_reations_1 = get_adj_relations(level_1, adj, mapping_value)
    level_1["adj"] = adj_reations_1

    level_2 = {}
    level_3 = {}
    contains_12 = {}
    contains_23 = {}
    # level_2 = {val: bm_num_dict[val] for val in np.unique(seg_map_stage_2)}
    # level_2 = {str(i): {'role': 'L2', 'words': data} for i, data in enumerate(level_2.values())}
    # adj_reations_2 = get_adj_relations(level_2, adj_stage_2, bm_num_dict, name_bm_dict)
    # level_2["adj"] = adj_reations_2
    #
    # level_3 = {val: bm_num_dict[val] for val in np.unique(seg_map_stage_3)}
    # level_3 = {str(i): {'role': 'L3', 'words': data} for i, data in enumerate(level_3.values())}
    # adj_reations_3 = get_adj_relations(level_3, adj_stage_3, bm_num_dict, name_bm_dict)
    # level_3["adj"] = adj_reations_3

    # {
    #   "caption":这是一个 [landcover / landuse] 分类图，其中一级类为[L1_1,L1_2,L1_3].二级类为[L2_1, L2_2, L3_3].三级类为[L3_1, L3_2, L3_3]

    #   'Level-1': {
    #     '0': {'role': 'L1',  'words': ['L1_1']},
    #     '1': {'role': 'L1',  'words': ['L1_2']},
    #     '2': {'role': 'L1',  'words': ['L1_3']},
    #     'adj':[[0,1], [1, 2]]
    #   },
    #   'Level-2': {
    #     '0': {'role': 'L2',  'words': ['L2_1']},
    #     '1': {'role': 'L2',  'words': ['L2_2']},
    #     '2': {'role': 'L2',  'words': ['L2_3']},
    #     'adj':[[0,1], [1, 2]]
    #   },
    #   'Level-3': {
    #     '0': {'role': 'L3',  'words': ['L2_1']},
    #     '1': {'role': 'L3',  'words': ['L2_2']},
    #     '2': {'role': 'L3',  'words': ['L2_3']},
    #     'adj':[[0,1], [1, 2]]
    #   },

    #   'contains_12': [[0, 1], [1, 1], [1, 2]]
    #   'contains_23': [[0, 1], [1, 1], [1, 2]]
    # }

    tag = "Open Earth Map"
    l1 = ', '.join([mapping_value[str(val)] for val in np.unique(seg_map)])
    l2 = 'None'
    l3 = 'None'

    captions = "This is a {} classification map, where the primary classes are [{}], the secondary classes are [{}], and the tertiary classes are [{}].".format(
      tag, l1, l2, l3)

    result = {
      "caption": captions,
      "Level-1": level_1,
      "Level-2": level_2,
      "Level-3": level_3,
      "contains_12": contains_12,
      "contains_23": contains_23
    }

    def convert_to_native_types(obj):
      if isinstance(obj, np.integer):
        return int(obj)
      elif isinstance(obj, np.floating):
        return float(obj)
      elif isinstance(obj, np.ndarray):
        return obj.tolist()
      elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
      elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
      return obj

    caption = convert_to_native_types(result)

    outname = os.path.join(out_dir, os.path.basename(segmap).replace('.tif', '.json'))

    with open(outname, 'w', encoding='utf-8') as outfile:
      json.dump(caption, outfile, indent=4)

    print(f"Saved {outname}")
def convert_semantic_to_captions(segmap, mapping_value, bm_num_dict, name_bm_dict, out_dir):
    seg_map = np.array(Image.open(segmap))

    seg_map_stage_1 = update_segMap(seg_map, mapping_value, name_bm_dict, bm_num_dict, landcover, 1)
    seg_map_stage_2 = update_segMap(seg_map, mapping_value, name_bm_dict, bm_num_dict, landcover, 2)
    seg_map_stage_3 = seg_map

    adj_stage_1 = calculate_adjacency_list_8_neighborhood(seg_map_stage_1)
    adj_stage_2 = calculate_adjacency_list_8_neighborhood(seg_map_stage_2)
    adj_stage_3 = calculate_adjacency_list_8_neighborhood(seg_map_stage_3)
    contains_12 = calculate_inclusion_relationship(seg_map_stage_1, seg_map_stage_2)
    contains_23 = calculate_inclusion_relationship(seg_map_stage_2, seg_map_stage_3)


    level_1 = {val: bm_num_dict[val] for val in np.unique(seg_map_stage_1)}
    level_1 =  {str(i): {'role': 'L1',  'words': data} for i, data in enumerate(level_1.values())}

    adj_reations_1 = get_adj_relations(level_1, adj_stage_1, bm_num_dict, name_bm_dict)
    level_1["adj"] = adj_reations_1

    level_2 = {val: bm_num_dict[val] for val in np.unique(seg_map_stage_2)}
    level_2 =  {str(i): {'role': 'L2',  'words': data} for i, data in enumerate(level_2.values())}
    adj_reations_2 = get_adj_relations(level_2, adj_stage_2, bm_num_dict, name_bm_dict)
    level_2["adj"] = adj_reations_2


    level_3 = {val: bm_num_dict[val] for val in np.unique(seg_map_stage_3)}
    level_3 =  {str(i): {'role': 'L3',  'words': data} for i, data in enumerate(level_3.values())}
    adj_reations_3 = get_adj_relations(level_3, adj_stage_3, bm_num_dict, name_bm_dict)
    level_3["adj"] = adj_reations_3










    # {
    #   "caption":这是一个 [landcover / landuse] 分类图，其中一级类为[L1_1,L1_2,L1_3].二级类为[L2_1, L2_2, L3_3].三级类为[L3_1, L3_2, L3_3]

    #   'Level-1': {
    #     '0': {'role': 'L1',  'words': ['L1_1']},
    #     '1': {'role': 'L1',  'words': ['L1_2']},
    #     '2': {'role': 'L1',  'words': ['L1_3']},
    #     'adj':[[0,1], [1, 2]]
    #   },
    #   'Level-2': {
    #     '0': {'role': 'L2',  'words': ['L2_1']},
    #     '1': {'role': 'L2',  'words': ['L2_2']},
    #     '2': {'role': 'L2',  'words': ['L2_3']},
    #     'adj':[[0,1], [1, 2]]
    #   },
    #   'Level-3': {
    #     '0': {'role': 'L3',  'words': ['L2_1']},
    #     '1': {'role': 'L3',  'words': ['L2_2']},
    #     '2': {'role': 'L3',  'words': ['L2_3']},
    #     'adj':[[0,1], [1, 2]]
    #   },

    #   'contains_12': [[0, 1], [1, 1], [1, 2]]
    #   'contains_23': [[0, 1], [1, 1], [1, 2]]
    # }

    tag = "landcover"
    l1 = ', '.join([ bm_num_dict[val]for val in np.unique(seg_map_stage_1) ])
    l2 = ', '.join([ bm_num_dict[val]for val in np.unique(seg_map_stage_2) ])
    l3 = ', '.join([ bm_num_dict[val]for val in np.unique(seg_map_stage_3) ])


    captions = "This is a {} classification map, where the primary classes are [{}], the secondary classes are [{}], and the tertiary classes are [{}].".format(tag, l1, l2, l3)



    result = {
      "caption": captions,
      "Level-1": level_1,
      "Level-2": level_2,
      "Level-3": level_3,
      "contains_12": contains_12,
      "contains_23": contains_23
    }

    def convert_to_native_types(obj):
      if isinstance(obj, np.integer):
        return int(obj)
      elif isinstance(obj, np.floating):
        return float(obj)
      elif isinstance(obj, np.ndarray):
        return obj.tolist()
      elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
      elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
      return obj
    caption = convert_to_native_types(result)

    outname = os.path.join(out_dir, os.path.basename(segmap).replace('.tif', '.json'))

    with open(outname, 'w', encoding='utf-8') as outfile:
      json.dump(caption, outfile, indent=4)

    print(f"Saved {outname}")


def crop_images( samples_path, crop_size=(256, 256), save_dir=None):

    # with open(txt, encoding='utf-8', mode='r') as f:
    #   lines = f.readlines()
    # samples_path = []
    # for line in lines:
    #   line = line.rstrip()
    #   if line in all_paths:
    #     samples_path.append(all_paths[line])
    #   else:
    #     print("Error: can not find path:{}".format(line))
    # image_paths = [path[0] for path in samples_path]
    # label_paths = [path[1] for path in samples_path]

    image_paths, label_paths = samples_path
    if save_dir is None:
      save_dir = os.path.join(save_dir, 'cropped')
    os.makedirs(save_dir, exist_ok=True)

    cropped_img_dir = os.path.join(save_dir, 'images')
    cropped_lbl_dir = os.path.join(save_dir, 'labels')
    os.makedirs(cropped_img_dir, exist_ok=True)
    os.makedirs(cropped_lbl_dir, exist_ok=True)

    for img_path, lbl_path in tqdm(zip(image_paths, label_paths)):
      with rasterio.open(img_path) as src_img:
        with rasterio.open(lbl_path) as src_lbl:
          # Loop through a grid of crops
          for i in range(0, src_img.width, crop_size[0]):
            for j in range(0, src_img.height, crop_size[1]):
              # Define the window position and size
              window = Window(i, j, crop_size[0], crop_size[1])
              # Read the window and its transform
              img_data = src_img.read(window=window)
              lbl_data = src_lbl.read(window=window)
              img_transform = src_img.window_transform(window)
              lbl_transform = src_lbl.window_transform(window)

              # Define file paths for saving
              cropped_img_name = f"{os.path.basename(img_path).replace('.tif', '')}_{i}_{j}.tif"
              cropped_lbl_name = f"{os.path.basename(lbl_path).replace('.tif', '')}_{i}_{j}.tif"

              # Write the cropped image and label with the updated transform
              with rasterio.open(
                      os.path.join(cropped_img_dir, cropped_img_name), 'w',
                      driver='GTiff', height=crop_size[1], width=crop_size[0],
                      count=src_img.count, dtype=src_img.dtypes[0],
                      crs=src_img.crs, transform=img_transform
              ) as dst_img:
                dst_img.write(img_data)

              with rasterio.open(
                      os.path.join(cropped_lbl_dir, cropped_lbl_name), 'w',
                      driver='GTiff', height=crop_size[1], width=crop_size[0],
                      count=src_lbl.count, dtype=src_lbl.dtypes[0],
                      crs=src_lbl.crs, transform=lbl_transform
              ) as dst_lbl:
                dst_lbl.write(lbl_data)


# Load pre-trained ResNet-50
def resnet_50():
  model = models.resnet50(pretrained=True)
  model.eval()  # Set the model to evaluation mode
  return model


# Preprocess the image
def preprocess_image(image_path):
  transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  image = Image.open(image_path).convert('RGB')
  return transform(image).unsqueeze(0)  # Add batch dimension


# Extract features from ResNet-50
def extract_features(image_path, model):
  image_tensor = preprocess_image(image_path)
  with torch.no_grad():  # Disable gradient calculation
    features = model(image_tensor)  # Forward pass through the model
  return features.flatten()  # Flatten the feature map


# Show the diversity of regions based on feature distributions
from sklearn.decomposition import PCA
import seaborn as sns
def show_oem_diversity(image_paths):
    pretrained_net = resnet_50()

    region_features = {}
    region_labels = []

    for path in tqdm(image_paths):
      # Assuming the image path contains region information separated by an underscore
      name = os.path.basename(path)
      region_name = name.split('_')[0]  # Extract region from the filename
      features = extract_features(path, pretrained_net)

      # Store features by region
      if region_name not in region_features:
        region_features[region_name] = []
      region_features[region_name].append(features)
      region_labels.append(region_name)

    # Visualize the feature distributions using scatter plot
    plt.figure(figsize=(12, 10))

    # Define a color map for the regions
    region_colors = {'Region1': 'r', 'Region2': 'g', 'Region3': 'b'}  # Example color mapping
    # for region, features in region_features.items():
    #   # Convert features to numpy for plotting
    #   features_np = np.array([f.numpy() for f in features])
    #   plt.scatter(features_np[:, 0], features_np[:, 1], label=region,
    #               marker='o', s=100)  # Use 'o' for circles with size 100
    # Initialize an empty list to store legend handles

    # Dynamically generate N distinct colors using seaborn color palette
    colors = sns.color_palette("Set2",
                               len(region_features))  # You can use other color palettes like 'Paired', 'Set1', etc.
    legend_handles = []
    for i,(region, features) in enumerate(region_features.items()):
      # Convert features to numpy for plotting
      #features_np = np.array([f.numpy() for f in features])

      features_np = np.array([f.numpy() for f in features])

      # Find the center of the feature distribution (mean)
      #center_feature = np.mean(features_np, axis=0)

      #Perform dimensionality reduction to 2D using PCA
      pca = PCA(n_components=2,random_state=42)
      reduced_features = pca.fit_transform(features_np)  # Apply PCA to all features in the region
      # # Optionally scale the reduced features for better spread
      scaled_center_feature = np.mean(reduced_features, axis=0) * 4  # Scale the center feature
      # Apply t-SNE to reduce the dimensionality and increase separation
      # tsne = TSNE(n_components=2, random_state=42)
      # reduced_features_tsne = tsne.fit_transform(features_np)  # Apply t-SNE to features
      # scaled_center_feature = np.mean(reduced_features, axis=0) * 4  # Scale the center feature
      #
      # # Calculate the number of features in the region
      num_features = len(features)

      # Set the initial size for the circle (base size 100)
      circle_size = num_features * 2.0  # Adjust the multiplier for desired circle size increase

      # Plot the center feature as a circle with dynamic size based on num_features
      scatter_center = plt.scatter(scaled_center_feature[0], scaled_center_feature[1],
                                   label=region, s=circle_size,
                                   alpha=0.6, color=colors[i])

      # Add the scatter center to the legend with a fixed size circle for consistency
      legend_handles.append(plt.scatter([], [], color=scatter_center.get_facecolor(),
                                        s=10, alpha=0.6, label=region))




    plt.title('Feature Distribution for Different Regions')
    #plt.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.15, 1), borderaxespad=0.)
    # Adjust the layout to make sure the plot fits well and everything is visible
    plt.tight_layout()
    plt.show()

def show_num_obj(image_paths):
    from skimage import measure
    num_objects_per_map = []
    # Iterate through each segmentation map
    for segmap_path in tqdm(image_paths):
      # Read the segmentation map (assuming it is grayscale and has labeled objects)
      segmap = np.array(Image.open(segmap_path))

      # Label the connected components (objects) in the segmentation map
      labeled_map = measure.label(segmap, connectivity=2)  # connectivity=2 ensures 8-connectivity
      num_objects = np.max(labeled_map)  # The maximum label value corresponds to the number of objects
      num_objects_per_map.append(num_objects)

    # Plot the distribution of objects in segmentation maps
    plt.figure(figsize=(10, 6))
    plt.hist(num_objects_per_map, bins=50, edgecolor='black', alpha=0.7)
    # Applying log scale on the y-axis to better visualize the distribution
    plt.yscale('log')

    # Adding labels and title with larger font sizes
    # Adding labels and title with larger font sizes, using Times New Roman
    plt.title("Distribution of Segmented Objects per Map", fontsize=14, fontname='Times New Roman')
    plt.xlabel("Number of Objects", fontsize=12, fontname='Times New Roman')
    plt.ylabel("Frequency (log scale)", fontsize=12, fontname='Times New Roman')

    # Enhancing the grid
    plt.grid(True, linestyle='--', alpha=0.5)

    # Saving and displaying the plot
    plt.tight_layout()
    plt.savefig("object_distribution_improved.png")
    plt.show()
def main(args, ext = "*.tif"):
    data_dir = args.data_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
#data_dir = r"F:\data\M2I\WV3058303058640_01_P00620180318F_8BitRGB_samples\WV3058303058640_01_P00620180318F_8BitRGB_label_2279.tif"

    img_dir = data_dir + "/images"
    lbl_dir = data_dir + "/labels"

    imgs = glob.glob(os.path.join(img_dir, ext))
    segmaps = glob.glob(os.path.join(lbl_dir, ext))

    #统计所有segmaps中分割对象的数量，并绘制

    show_oem_diversity(imgs)

    sys.exit(-1)
    #crop to [256,256]
    # save_dir = data_dir + "/256"
    # os.makedirs(save_dir, exist_ok=True)
    # crop_images([imgs, segmaps],crop_size=(256,256),save_dir = save_dir)

    from ldm.modules.encoders.modules import FrozenCLIPTextEmbedder
    clip_model = FrozenCLIPTextEmbedder().cuda().eval()
    node_dict = get_node_feats(clip_model, oem["l1"])


    # clip_model, _ = clip.load("./pretrained/clip_checkpoints/ViT-B-16.pt", "cuda")
    # clip_model = clip_model.to("cuda").eval()
    # for p in clip_model.parameters():
    #   p.requires_grad = False
    #clip_model = None


    if args.tag == "landcover":

      xml = args.transxml
      mapping_value = get_mapping_value(xml)
      bm_num_dict = get_caps_mapping_val(landcover,xml) #{"0100":0}
      name_bm_dict = generate_dict(landcover)#{"0100":Farmland}

      NUM_THREADS = 4
      Parallel(n_jobs=NUM_THREADS)(delayed(convert_semantic_to_captions)(t, mapping_value,bm_num_dict,name_bm_dict, out_dir ) for t in tqdm(segmaps))
    else:
      mapping_value = oem["l1"]
      NUM_THREADS = 7

      # for t in tqdm(segmaps):
      #     convert_oem_to_captions(t, mapping_value, out_dir)
      Parallel(n_jobs=NUM_THREADS)(delayed(convert_oem_to_captions)(node_dict, t, mapping_value, out_dir)for t in tqdm(segmaps))






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transxml", type=str)
    parser.add_argument("--prompts", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--tag", type=str, choices=["landcover", "oem"],default= 'landcover')
    args = parser.parse_args()
    main(args)


