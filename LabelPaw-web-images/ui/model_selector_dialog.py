import os
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, 
                               QListWidget, QListWidgetItem, QWidget, QLabel, QScrollArea, QFrame)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QFont, QCursor, QPainter, QColor, QPixmap

class ModelItemWidget(QWidget):
    item_clicked = Signal(dict)

    def __init__(self, model_info, is_selected=False, is_dark_theme=False, parent=None):
        super().__init__(parent)
        self.model_info = model_info
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        
        self.icon_label = QLabel()
        if is_selected:
            icon_path = "ui/icon/打勾1.svg"
            pixmap = QIcon(icon_path).pixmap(16, 16)
            
            # If dark theme, tint the icon to white/light gray
            if is_dark_theme:
                painter = QPainter(pixmap)
                painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
                painter.fillRect(pixmap.rect(), QColor("#F8FAFC"))
                painter.end()
                
            self.icon_label.setPixmap(pixmap)
        else:
            self.icon_label.setFixedSize(16, 16)
        
        text_color = "#F8FAFC" if is_dark_theme else "#333333"
        dim_text = "#94A3B8" if is_dark_theme else "#888888"
        hover_bg = "#334155" if is_dark_theme else "#F3F4F6"

        self.name_label = QLabel(model_info.get("display_name", "Unknown Model"))
        self.name_label.setStyleSheet(f"font-size: 14px; color: {text_color};")
        
        self.size_label = QLabel(model_info.get("size_label", ""))
        self.size_label.setStyleSheet(f"font-size: 12px; color: {dim_text};")
        self.size_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        layout.addWidget(self.icon_label)
        layout.addWidget(self.name_label, 1)
        layout.addWidget(self.size_label)
        
        self.setStyleSheet(f"""
            QWidget {{
                background-color: transparent;
                border-radius: 6px;
            }}
            QWidget:hover {{
                background-color: {hover_bg};
            }}
        """)
        self.setCursor(QCursor(Qt.PointingHandCursor))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.item_clicked.emit(self.model_info)

class ModelSelectorDialog(QDialog):
    model_selected = Signal(dict)

    def __init__(self, current_model_key=None, is_dark_theme=False, parent=None):
        super().__init__(parent)
        self.current_model_key = current_model_key
        self.is_dark_theme = is_dark_theme
        
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(500, 400)
        
        # Theme colors
        bg_color = "#1E293B" if is_dark_theme else "#FFFFFF"
        border_color = "#334155" if is_dark_theme else "#E2E8F0"
        text_color = "#F8FAFC" if is_dark_theme else "#334155"
        list_bg = bg_color if is_dark_theme else "#F8FAFC"
        list_item_text = "#94A3B8" if is_dark_theme else "#64748B"
        list_item_selected_bg = "#1E293B" if is_dark_theme else "#FFFFFF"
        list_item_selected_text = "#F8FAFC" if is_dark_theme else "#0F172A"
        list_item_hover_bg = "#334155" if is_dark_theme else "#F1F5F9"
        
        # Main Container
        self.main_container = QFrame(self)
        self.main_container.setObjectName("MainContainer")
        self.main_container.setStyleSheet(f"""
            #MainContainer {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 12px;
            }}
        """)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.main_container)
        
        container_layout = QVBoxLayout(self.main_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        
        # Search Box
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索模型...")
        self.search_input.setStyleSheet(f"""
            QLineEdit {{
                border: none;
                border-bottom: 1px solid {border_color};
                padding: 12px 16px;
                font-size: 14px;
                color: {text_color};
                background-color: transparent;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
            }}
            QLineEdit:focus {{
                outline: none;
            }}
        """)
        container_layout.addWidget(self.search_input)
        
        # Content Area
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Left Categories
        self.category_list = QListWidget()
        self.category_list.setFixedWidth(160)
        self.category_list.setStyleSheet(f"""
            QListWidget {{
                border: none;
                border-right: 1px solid {border_color};
                background-color: {list_bg};
                outline: none;
                border-bottom-left-radius: 12px;
            }}
            QListWidget:focus {{
                outline: none;
            }}
            QListWidget::item {{
                padding: 12px 16px;
                color: {list_item_text};
                font-weight: bold;
                border: none;
                outline: none;
            }}
            QListWidget::item:focus {{
                outline: none;
                border: none;
            }}
            QListWidget::item:selected {{
                background-color: {list_item_selected_bg};
                color: {list_item_selected_text};
                border: none;
                outline: none;
            }}
            QListWidget::item:hover:!selected {{
                background-color: {list_item_hover_bg};
            }}
        """)
        
        # Right Models Area
        self.models_area = QScrollArea()
        self.models_area.setWidgetResizable(True)
        self.models_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {bg_color};
                border-bottom-right-radius: 12px;
            }}
            QWidget#ModelsContainer {{
                background-color: {bg_color};
            }}
        """)
        
        self.models_container = QWidget()
        self.models_container.setObjectName("ModelsContainer")
        self.models_layout = QVBoxLayout(self.models_container)
        self.models_layout.setContentsMargins(10, 10, 10, 10)
        self.models_layout.setSpacing(2)
        self.models_layout.addStretch()
        self.models_area.setWidget(self.models_container)
        
        content_layout.addWidget(self.category_list)
        content_layout.addWidget(self.models_area)
        
        container_layout.addLayout(content_layout)
        
        # Signals
        self.category_list.currentRowChanged.connect(self._on_category_changed)
        self.search_input.textChanged.connect(self._on_search)
        self.model_selected.connect(self.accept)

        self._init_data()
    def _init_data(self):
        # 设置一个更加通用的模型权重基准目录
        # 优先从当前项目根目录下的 weights 文件夹寻找，如果没有则回退到硬编码路径
        import sys
        if getattr(sys, 'frozen', False):
            PROJECT_ROOT = os.path.dirname(sys.executable)
        else:
            PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        LOCAL_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
        HARDCODED_DEV_DIR = r"E:\11-AI\标注工具\weights"

        if os.path.exists(LOCAL_WEIGHTS_DIR):
            MODEL_BASE_DIR = LOCAL_WEIGHTS_DIR
        elif os.path.exists(HARDCODED_DEV_DIR):
            MODEL_BASE_DIR = HARDCODED_DEV_DIR
        else:
            MODEL_BASE_DIR = LOCAL_WEIGHTS_DIR  # 默认指向项目下的 weights
        
        # Initialize all_data
        self.all_data = {}
        
        # Load SAM models from SAM_MODEL_MAP (dynamically scanned in sam_client)
        try:
            from labelpaw.models.sam_client import SAM_MODEL_MAP
            if SAM_MODEL_MAP:
                sam_list = []
                for key, info in SAM_MODEL_MAP.items():
                    sam_list.append({
                        "key": key,
                        "display_name": info.get("display_name", key),
                        "size_label": info.get("size_label", ""),
                        "type": info.get("type", "sam2"),
                        "weight": info.get("weight", ""),
                        "supports_text": info.get("supports_text", False)
                    })
                if sam_list:
                    self.all_data["SAM"] = sam_list
        except ImportError:
            pass
        
        # Scan for YOLO weights directories dynamically
        if os.path.exists(MODEL_BASE_DIR):
            for item in os.listdir(MODEL_BASE_DIR):
                item_path = os.path.join(MODEL_BASE_DIR, item)
                if os.path.isdir(item_path) and item.startswith("yolo") and item.endswith("_weights"):
                    # Extract the version prefix, e.g., "yolo26", "yolov8"
                    version_prefix = item.replace("_weights", "")
                    category_name = version_prefix.upper()
                    
                    local_yolo = []
                    for f in os.listdir(item_path):
                        if f.endswith(('.pt', '.onnx')):
                            file_size_mb = os.path.getsize(os.path.join(item_path, f)) // (1024 * 1024)
                            local_yolo.append({
                                "key": f.replace('.pt', '').replace('.onnx', ''),
                                "display_name": f.replace('.pt', '').replace('.onnx', ''),
                                "size_label": f"{file_size_mb} MB",
                                "type": version_prefix,
                                "path": os.path.join(item_path, f)
                            })
                    if local_yolo:
                        self.all_data[category_name] = local_yolo

        # Populate categories
        for cat in self.all_data.keys():
            self.category_list.addItem(cat)
            
        target_row = 0
        if self.current_model_key:
            for idx, (cat, models) in enumerate(self.all_data.items()):
                if any(m["key"] == self.current_model_key for m in models):
                    target_row = idx
                    break

        if self.category_list.count() > 0:
            self.category_list.setCurrentRow(target_row)

    def _on_category_changed(self, row):
        # When category changes, re-evaluate with current search text
        self._update_models_list(search_text=self.search_input.text().lower())

    def _on_search(self, text):
        self._update_models_list(search_text=text.lower())

    def _update_models_list(self, search_text=""):
        # Clear existing models
        for i in reversed(range(self.models_layout.count() - 1)): # keep the stretch
            widget_to_remove = self.models_layout.itemAt(i).widget()
            self.models_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)
            
        if search_text:
            # Global search across all categories
            models = []
            for cat_models in self.all_data.values():
                models.extend(cat_models)
        else:
            # Show only current category
            current_cat = self.category_list.currentItem()
            if not current_cat:
                return
            cat_text = current_cat.text()
            models = self.all_data.get(cat_text, [])
        
        # Keep track of added keys to prevent duplicates in global search if they exist
        added_keys = set()
        
        for model in models:
            if search_text and search_text not in model.get("display_name", "").lower() and search_text not in model.get("key", "").lower():
                continue
                
            if search_text and model["key"] in added_keys:
                continue
                
            added_keys.add(model["key"])
                
            is_selected = (model["key"] == self.current_model_key)
            item_widget = ModelItemWidget(model, is_selected=is_selected, is_dark_theme=self.is_dark_theme, parent=self.models_container)
            item_widget.item_clicked.connect(self.model_selected.emit)
            self.models_layout.insertWidget(self.models_layout.count() - 1, item_widget)
