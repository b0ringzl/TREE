import json
import os

def normalize_template(template):
    keypoints = template.get("keypoints", [])
    if not keypoints:
        return template
    xs = [kp["default_pos"][0] for kp in keypoints]
    ys = [kp["default_pos"][1] for kp in keypoints]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    span_x = max_x - min_x
    span_y = max_y - min_y
    
    target_min_x = 0.144
    target_max_x = 0.856
    target_min_y = 0.096
    target_max_y = 0.904
    
    target_span_x = target_max_x - target_min_x  # 0.712
    target_span_y = target_max_y - target_min_y  # 0.808
    
    for kp in keypoints:
        x, y = kp["default_pos"]
        
        # Normalize to target bounds
        if span_x > 0:
            new_x = target_min_x + ((x - min_x) / span_x) * target_span_x
        else:
            new_x = 0.5
            
        if span_y > 0:
            new_y = target_min_y + ((y - min_y) / span_y) * target_span_y
        else:
            new_y = 0.5
            
        kp["default_pos"] = [round(new_x, 5), round(new_y, 5)]
        
    return template

DEFAULT_TEMPLATES = [
    {
        "name": "Person (COCO)",
        "label": "person",
        "description": "COCO human body pose estimation",
        "kpt_shape": [17, 3],
        "keypoints": [
            {"name": "nose", "color": "#FF0000", "default_pos": [0.5, 0.11212]},
            {"name": "left_eye", "color": "#FF5500", "default_pos": [0.45011, 0.09347]},
            {"name": "right_eye", "color": "#FF0055", "default_pos": [0.54978, 0.09347]},
            {"name": "left_ear", "color": "#FFAA00", "default_pos": [0.39978, 0.11212]},
            {"name": "right_ear", "color": "#FF00AA", "default_pos": [0.60022, 0.11212]},
            {"name": "left_shoulder", "color": "#FFFF00", "default_pos": [0.31233, 0.25224]},
            {"name": "right_shoulder", "color": "#AAFF00", "default_pos": [0.68767, 0.25224]},
            {"name": "left_elbow", "color": "#55FF00", "default_pos": [0.18767, 0.41120]},
            {"name": "right_elbow", "color": "#00FF00", "default_pos": [0.81222, 0.41120]},
            {"name": "left_wrist", "color": "#00FF55", "default_pos": [0.12511, 0.53272]},
            {"name": "right_wrist", "color": "#00FFAA", "default_pos": [0.87489, 0.53272]},
            {"name": "left_hip", "color": "#00FFFF", "default_pos": [0.37489, 0.53272]},
            {"name": "right_hip", "color": "#00AAFF", "default_pos": [0.62511, 0.53272]},
            {"name": "left_knee", "color": "#0055FF", "default_pos": [0.34989, 0.71971]},
            {"name": "right_knee", "color": "#0000FF", "default_pos": [0.65, 0.71971]},
            {"name": "left_ankle", "color": "#5500FF", "default_pos": [0.325, 0.90671]},
            {"name": "right_ankle", "color": "#AA00FF", "default_pos": [0.67489, 0.90671]}
        ],
        "connections": [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
            [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
        ]
    },
    {
        "name": "Hand",
        "label": "hand",
        "description": "Hand Keypoints (21 points)",
        "kpt_shape": [21, 3],
        "keypoints": [
            {"name": "wrist", "color": "#FF0000", "default_pos": [0.5, 0.9]},
            {"name": "thumb_cmc", "color": "#FFA500", "default_pos": [0.4, 0.75]},
            {"name": "thumb_mcp", "color": "#FFA500", "default_pos": [0.3, 0.6]},
            {"name": "thumb_ip", "color": "#FFA500", "default_pos": [0.2, 0.5]},
            {"name": "thumb_tip", "color": "#FFA500", "default_pos": [0.1, 0.4]},
            {"name": "index_mcp", "color": "#FFFF00", "default_pos": [0.45, 0.55]},
            {"name": "index_pip", "color": "#FFFF00", "default_pos": [0.42, 0.35]},
            {"name": "index_dip", "color": "#FFFF00", "default_pos": [0.4, 0.2]},
            {"name": "index_tip", "color": "#FFFF00", "default_pos": [0.38, 0.1]},
            {"name": "middle_mcp", "color": "#00FF00", "default_pos": [0.55, 0.53]},
            {"name": "middle_pip", "color": "#00FF00", "default_pos": [0.55, 0.32]},
            {"name": "middle_dip", "color": "#00FF00", "default_pos": [0.55, 0.18]},
            {"name": "middle_tip", "color": "#00FF00", "default_pos": [0.55, 0.08]},
            {"name": "ring_mcp", "color": "#00FFFF", "default_pos": [0.65, 0.56]},
            {"name": "ring_pip", "color": "#00FFFF", "default_pos": [0.68, 0.38]},
            {"name": "ring_dip", "color": "#00FFFF", "default_pos": [0.7, 0.25]},
            {"name": "ring_tip", "color": "#00FFFF", "default_pos": [0.72, 0.15]},
            {"name": "pinky_mcp", "color": "#0000FF", "default_pos": [0.75, 0.65]},
            {"name": "pinky_pip", "color": "#0000FF", "default_pos": [0.8, 0.5]},
            {"name": "pinky_dip", "color": "#0000FF", "default_pos": [0.83, 0.4]},
            {"name": "pinky_tip", "color": "#0000FF", "default_pos": [0.85, 0.3]}
        ],
        "connections": [
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [0, 9], [9, 10], [10, 11], [11, 12],
            [0, 13], [13, 14], [14, 15], [15, 16],
            [0, 17], [17, 18], [18, 19], [19, 20],
            [5, 9], [9, 13], [13, 17]
        ]
    },
{
            "name": "Face (68 pts)",
            "label": "face",
            "description": "iBUG 300W Face Landmarks",
            "kpt_shape": [
                68,
                3
            ],
            "keypoints": [
                {
                    "name": "pt_0",
                    "color": "#FFA500",
                    "default_pos": [
                        0.15,
                        0.7
                    ]
                },
                {
                    "name": "pt_1",
                    "color": "#FFA500",
                    "default_pos": [
                        0.16,
                        0.62
                    ]
                },
                {
                    "name": "pt_2",
                    "color": "#FFA500",
                    "default_pos": [
                        0.18,
                        0.54
                    ]
                },
                {
                    "name": "pt_3",
                    "color": "#FFFF00",
                    "default_pos": [
                        0.21,
                        0.46
                    ]
                },
                {
                    "name": "pt_4",
                    "color": "#FF69B4",
                    "default_pos": [
                        0.25,
                        0.38
                    ]
                },
                {
                    "name": "pt_5",
                    "color": "#87CEFA",
                    "default_pos": [
                        0.3,
                        0.31
                    ]
                },
                {
                    "name": "pt_6",
                    "color": "#87CEFA",
                    "default_pos": [
                        0.36,
                        0.25
                    ]
                },
                {
                    "name": "pt_7",
                    "color": "#FF69B4",
                    "default_pos": [
                        0.43,
                        0.2
                    ]
                },
                {
                    "name": "pt_8",
                    "color": "#87CEFA",
                    "default_pos": [
                        0.5,
                        0.17
                    ]
                },
                {
                    "name": "pt_9",
                    "color": "#87CEFA",
                    "default_pos": [
                        0.57,
                        0.2
                    ]
                },
                {
                    "name": "pt_10",
                    "color": "#FF69B4",
                    "default_pos": [
                        0.64,
                        0.25
                    ]
                },
                {
                    "name": "pt_11",
                    "color": "#FF0000",
                    "default_pos": [
                        0.7,
                        0.31
                    ]
                },
                {
                    "name": "pt_12",
                    "color": "#FF0000",
                    "default_pos": [
                        0.75,
                        0.38
                    ]
                },
                {
                    "name": "pt_13",
                    "color": "#FF0000",
                    "default_pos": [
                        0.79,
                        0.46
                    ]
                },
                {
                    "name": "pt_14",
                    "color": "#32CD32",
                    "default_pos": [
                        0.82,
                        0.54
                    ]
                },
                {
                    "name": "pt_15",
                    "color": "#32CD32",
                    "default_pos": [
                        0.84,
                        0.62
                    ]
                },
                {
                    "name": "pt_16",
                    "color": "#32CD32",
                    "default_pos": [
                        0.85,
                        0.7
                    ]
                },
                {
                    "name": "pt_17",
                    "color": "#0000FF",
                    "default_pos": [
                        0.22,
                        0.82
                    ]
                },
                {
                    "name": "pt_18",
                    "color": "#FF0000",
                    "default_pos": [
                        0.27,
                        0.85
                    ]
                },
                {
                    "name": "pt_19",
                    "color": "#FFFFFF",
                    "default_pos": [
                        0.33,
                        0.86
                    ]
                },
                {
                    "name": "pt_20",
                    "color": "#FFA500",
                    "default_pos": [
                        0.39,
                        0.85
                    ]
                },
                {
                    "name": "pt_21",
                    "color": "#FFA500",
                    "default_pos": [
                        0.44,
                        0.82
                    ]
                },
                {
                    "name": "pt_22",
                    "color": "#FFA07A",
                    "default_pos": [
                        0.56,
                        0.82
                    ]
                },
                {
                    "name": "pt_23",
                    "color": "#FFFF00",
                    "default_pos": [
                        0.61,
                        0.85
                    ]
                },
                {
                    "name": "pt_24",
                    "color": "#FF69B4",
                    "default_pos": [
                        0.67,
                        0.86
                    ]
                },
                {
                    "name": "pt_25",
                    "color": "#87CEFA",
                    "default_pos": [
                        0.73,
                        0.85
                    ]
                },
                {
                    "name": "pt_26",
                    "color": "#87CEFA",
                    "default_pos": [
                        0.78,
                        0.82
                    ]
                },
                {
                    "name": "pt_27",
                    "color": "#FF69B4",
                    "default_pos": [
                        0.5,
                        0.75
                    ]
                },
                {
                    "name": "pt_28",
                    "color": "#1E90FF",
                    "default_pos": [
                        0.5,
                        0.67
                    ]
                },
                {
                    "name": "pt_29",
                    "color": "#1E90FF",
                    "default_pos": [
                        0.5,
                        0.59
                    ]
                },
                {
                    "name": "pt_30",
                    "color": "#FF0000",
                    "default_pos": [
                        0.5,
                        0.51
                    ]
                },
                {
                    "name": "pt_31",
                    "color": "#FF0000",
                    "default_pos": [
                        0.42,
                        0.45
                    ]
                },
                {
                    "name": "pt_32",
                    "color": "#FF0000",
                    "default_pos": [
                        0.46,
                        0.43
                    ]
                },
                {
                    "name": "pt_33",
                    "color": "#90EE90",
                    "default_pos": [
                        0.5,
                        0.42
                    ]
                },
                {
                    "name": "pt_34",
                    "color": "#90EE90",
                    "default_pos": [
                        0.54,
                        0.43
                    ]
                },
                {
                    "name": "pt_35",
                    "color": "#90EE90",
                    "default_pos": [
                        0.58,
                        0.45
                    ]
                },
                {
                    "name": "pt_36",
                    "color": "#32CD32",
                    "default_pos": [
                        0.28,
                        0.73
                    ]
                },
                {
                    "name": "pt_37",
                    "color": "#0000FF",
                    "default_pos": [
                        0.32,
                        0.75
                    ]
                },
                {
                    "name": "pt_38",
                    "color": "#FF0000",
                    "default_pos": [
                        0.38,
                        0.75
                    ]
                },
                {
                    "name": "pt_39",
                    "color": "#FFFFFF",
                    "default_pos": [
                        0.42,
                        0.72
                    ]
                },
                {
                    "name": "pt_40",
                    "color": "#FFA500",
                    "default_pos": [
                        0.38,
                        0.7
                    ]
                },
                {
                    "name": "pt_41",
                    "color": "#FFA500",
                    "default_pos": [
                        0.32,
                        0.7
                    ]
                },
                {
                    "name": "pt_42",
                    "color": "#FFA500",
                    "default_pos": [
                        0.58,
                        0.72
                    ]
                },
                {
                    "name": "pt_43",
                    "color": "#FFFF00",
                    "default_pos": [
                        0.62,
                        0.75
                    ]
                },
                {
                    "name": "pt_44",
                    "color": "#FF69B4",
                    "default_pos": [
                        0.68,
                        0.75
                    ]
                },
                {
                    "name": "pt_45",
                    "color": "#87CEFA",
                    "default_pos": [
                        0.72,
                        0.73
                    ]
                },
                {
                    "name": "pt_46",
                    "color": "#FF69B4",
                    "default_pos": [
                        0.68,
                        0.7
                    ]
                },
                {
                    "name": "pt_47",
                    "color": "#FFA500",
                    "default_pos": [
                        0.62,
                        0.7
                    ]
                },
                {
                    "name": "pt_48",
                    "color": "#87CEFA",
                    "default_pos": [
                        0.35,
                        0.35
                    ]
                },
                {
                    "name": "pt_49",
                    "color": "#FFB6C1",
                    "default_pos": [
                        0.41,
                        0.38
                    ]
                },
                {
                    "name": "pt_50",
                    "color": "#FF0000",
                    "default_pos": [
                        0.47,
                        0.4
                    ]
                },
                {
                    "name": "pt_51",
                    "color": "#FF0000",
                    "default_pos": [
                        0.5,
                        0.41
                    ]
                },
                {
                    "name": "pt_52",
                    "color": "#FFFF00",
                    "default_pos": [
                        0.53,
                        0.4
                    ]
                },
                {
                    "name": "pt_53",
                    "color": "#90EE90",
                    "default_pos": [
                        0.59,
                        0.38
                    ]
                },
                {
                    "name": "pt_54",
                    "color": "#32CD32",
                    "default_pos": [
                        0.65,
                        0.35
                    ]
                },
                {
                    "name": "pt_55",
                    "color": "#32CD32",
                    "default_pos": [
                        0.59,
                        0.3
                    ]
                },
                {
                    "name": "pt_56",
                    "color": "#0000FF",
                    "default_pos": [
                        0.53,
                        0.28
                    ]
                },
                {
                    "name": "pt_57",
                    "color": "#FF0000",
                    "default_pos": [
                        0.5,
                        0.27
                    ]
                },
                {
                    "name": "pt_58",
                    "color": "#FFFFFF",
                    "default_pos": [
                        0.47,
                        0.28
                    ]
                },
                {
                    "name": "pt_59",
                    "color": "#87CEFA",
                    "default_pos": [
                        0.41,
                        0.3
                    ]
                },
                {
                    "name": "pt_60",
                    "color": "#FFA500",
                    "default_pos": [
                        0.38,
                        0.35
                    ]
                },
                {
                    "name": "pt_61",
                    "color": "#FFA500",
                    "default_pos": [
                        0.44,
                        0.37
                    ]
                },
                {
                    "name": "pt_62",
                    "color": "#FFA500",
                    "default_pos": [
                        0.5,
                        0.38
                    ]
                },
                {
                    "name": "pt_63",
                    "color": "#FFFF00",
                    "default_pos": [
                        0.56,
                        0.37
                    ]
                },
                {
                    "name": "pt_64",
                    "color": "#FF69B4",
                    "default_pos": [
                        0.62,
                        0.35
                    ]
                },
                {
                    "name": "pt_65",
                    "color": "#87CEFA",
                    "default_pos": [
                        0.56,
                        0.33
                    ]
                },
                {
                    "name": "pt_66",
                    "color": "#0000FF",
                    "default_pos": [
                        0.5,
                        0.32
                    ]
                },
                {
                    "name": "pt_67",
                    "color": "#FF0000",
                    "default_pos": [
                        0.44,
                        0.33
                    ]
                }
            ],
            "connections": [
                [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
                [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
                [15, 16], [17, 18], [18, 19], [19, 20], [20, 21], [22, 23], [23, 24],
                [24, 25], [25, 26], [27, 28], [28, 29], [29, 30], [30, 33], [31, 32],
                [32, 33], [33, 34], [34, 35], [36, 37], [37, 38], [38, 39], [39, 40],
                [40, 41], [41, 36], [42, 43], [43, 44], [44, 45], [45, 46], [46, 47],
                [47, 42], [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54],
                [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 48], [60, 61],
                [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 60]
            ]
    },
    {
        "name": "Rectangle",
        "label": "rectangle",
        "description": "4-point rectangle template",
        "kpt_shape": [4, 3],
        "keypoints": [
            {"name": "top_left", "color": "#FF0000", "default_pos": [0.25, 0.25]},
            {"name": "top_right", "color": "#00FF00", "default_pos": [0.75, 0.25]},
            {"name": "bottom_right", "color": "#0000FF", "default_pos": [0.75, 0.75]},
            {"name": "bottom_left", "color": "#FFFF00", "default_pos": [0.25, 0.75]}
        ],
        "connections": [
            [0, 1], [1, 2], [2, 3], [3, 0]
        ]
    },
    {
        "name": "Triangle",
        "label": "triangle",
        "description": "3-point triangle template",
        "kpt_shape": [3, 3],
        "keypoints": [
            {"name": "top", "color": "#FF0000", "default_pos": [0.5, 0.25]},
            {"name": "bottom_right", "color": "#00FF00", "default_pos": [0.75, 0.75]},
            {"name": "bottom_left", "color": "#0000FF", "default_pos": [0.25, 0.75]}
        ],
        "connections": [
            [0, 1], [1, 2], [2, 0]
        ]
    }
]

# Run dynamic normalization on all default templates to ensure strict conformance
for _t in DEFAULT_TEMPLATES:
    normalize_template(_t)

class TemplateManager:
    def __init__(self, config_dir="labelpaw/config/config"):
        self.config_dir = config_dir
        self.templates = []
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)
        self.load()

    def load(self):
        self.templates = []
        if os.path.exists(self.config_dir):
            for filename in os.listdir(self.config_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.config_dir, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            t = json.load(f)
                            t = normalize_template(t)
                            self.templates.append(t)
                    except Exception as e:
                        print(f"加载模板失败 {filename}: {e}")
        
        if not self.templates:
            self.templates = [normalize_template(t.copy()) for t in DEFAULT_TEMPLATES]
            self.save_all()

    def save_all(self):
        for t in self.templates:
            self.save_single(t)

    def save_single(self, template):
        name = template.get("name", "unknown").lower().replace(" ", "_")
        safe_name = "".join([c for c in name if c.isalnum() or c in ("_", "-")])
        filepath = os.path.join(self.config_dir, f"{safe_name}.json")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(template, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"保存模板失败 {safe_name}: {e}")

    def save(self):
        # 兼容旧代码对 save() 的调用
        self.save_all()

    def add_template(self, template):
        template = normalize_template(template)
        for i, t in enumerate(self.templates):
            if t["name"] == template["name"]:
                self.templates[i] = template
                self.save_single(template)
                return
        self.templates.append(template)
        self.save_single(template)

    def get_template_names(self):
        return [t["name"] for t in self.templates]

    def get_template(self, name):
        for t in self.templates:
            if t["name"] == name:
                return t
        return None

    def delete_template(self, name):
        for i, t in enumerate(self.templates):
            if t["name"] == name:
                # Remove file
                t_name = t.get("name", "unknown").lower().replace(" ", "_")
                safe_name = "".join([c for c in t_name if c.isalnum() or c in ("_", "-")])
                filepath = os.path.join(self.config_dir, f"{safe_name}.json")
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        print(f"删除模板文件失败 {filepath}: {e}")
                self.templates.pop(i)
                return True
        return False
