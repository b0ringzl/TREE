# -*- coding: utf-8 -*-
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, 
                               QListWidget, QLabel, QListWidgetItem)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCursor

class LabelDialog(QDialog):
    def __init__(self, title, prompt_text, class_list, default_label="", is_dark_theme=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.class_list = class_list
        self.is_dark_theme = is_dark_theme
        
        # UI Settings
        self.bg_color = "#1E293B" if is_dark_theme else "#FFFFFF"
        self.text_color = "#F8FAFC" if is_dark_theme else "#333333"
        self.input_bg = "#0F172A" if is_dark_theme else "#F8FAFC"
        self.input_border = "#334155" if is_dark_theme else "#E2E8F0"
        self.btn_bg = "#3B82F6" # Blue
        self.btn_text = "#FFFFFF"
        self.list_bg = self.input_bg
        self.list_hover = "#334155" if is_dark_theme else "#F1F5F9"
        
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowSystemMenuHint | Qt.WindowCloseButtonHint)
        self.setStyleSheet(f"""
            QDialog {{ background-color: {self.bg_color}; }}
            QLabel {{ color: {self.text_color}; font-size: 14px; font-weight: bold; }}
            QLineEdit {{
                background-color: {self.input_bg};
                color: {self.text_color};
                border: 1px solid {self.input_border};
                border-radius: 6px;
                padding: 10px;
                font-size: 14px;
            }}
            QLineEdit:focus {{
                border: 1px solid {self.btn_bg};
            }}
            QPushButton#mainBtn {{
                background-color: {self.btn_bg};
                color: {self.btn_text};
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton#mainBtn:hover {{ background-color: #2563EB; }}
            QPushButton#cancelBtn {{
                background-color: transparent;
                color: {self.text_color};
                border: 1px solid {self.input_border};
                border-radius: 6px;
                padding: 8px 20px;
                font-size: 14px;
            }}
            QPushButton#cancelBtn:hover {{ background-color: {self.input_border}; }}
            QPushButton#toggleBtn {{
                background-color: transparent;
                color: {self.btn_bg};
                border: none;
                font-size: 13px;
                text-align: left;
                padding: 5px 0px;
            }}
            QPushButton#toggleBtn:hover {{ text-decoration: underline; }}
            QListWidget {{
                background-color: {self.list_bg};
                color: {self.text_color};
                border: 1px solid {self.input_border};
                border-radius: 6px;
                outline: none;
            }}
            QListWidget:focus {{ outline: none; }}
            QListWidget::item {{
                padding: 8px;
                border: none;
                border-bottom: 1px solid {self.input_border};
            }}
            QListWidget::item:last {{
                border-bottom: none;
            }}
            QListWidget::item:focus {{ outline: none; border: none; border-bottom: 1px solid {self.input_border}; }}
            QListWidget::item:last:focus {{ border-bottom: none; }}
            QListWidget::item:hover {{
                background-color: {self.list_hover};
            }}
            QListWidget::item:selected {{
                background-color: {self.btn_bg};
                color: {self.btn_text};
            }}
            QScrollBar:vertical {{
                border: none;
                background: transparent;
                width: 8px;
                margin: 0px 0px 0px 0px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {self.input_border};
                min-height: 20px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: #64748B;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
        """)
        
        self.setMinimumWidth(360)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Prompt Label
        self.prompt_label = QLabel(prompt_text)
        layout.addWidget(self.prompt_label)
        
        # Input
        self.input_edit = QLineEdit()
        self.input_edit.setText(default_label)
        QTimer.singleShot(0, self.input_edit.selectAll)
        layout.addWidget(self.input_edit)
        
        # Toggle Button for List
        self.toggle_btn = QPushButton("▼ 展开历史类别列表" if class_list else "没有历史类别")
        self.toggle_btn.setObjectName("toggleBtn")
        self.toggle_btn.setCursor(QCursor(Qt.PointingHandCursor))
        if not class_list:
            self.toggle_btn.setEnabled(False)
            self.toggle_btn.setStyleSheet("color: gray;")
        else:
            self.toggle_btn.clicked.connect(self.toggle_list)
        layout.addWidget(self.toggle_btn)
        
        # List Widget (collapsible)
        self.list_widget = QListWidget()
        # Filter out any empty class names
        filtered_classes = [c for c in self.class_list if c.strip()]
        self.list_widget.addItems(filtered_classes)
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        # Limit height to approximately 5-6 items, forcing a scrollbar if there are more
        self.list_widget.setMaximumHeight(180)
        # Select matching item
        items = self.list_widget.findItems(default_label, Qt.MatchExactly)
        if items:
            self.list_widget.setCurrentItem(items[0])
            
        self.list_expanded = True if class_list else False
        self.list_widget.setVisible(self.list_expanded)
        layout.addWidget(self.list_widget)
        
        # Update toggle button text initially
        if self.list_expanded:
            self.toggle_btn.setText("▲ 收起历史类别列表")
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setObjectName("cancelBtn")
        self.btn_cancel.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_cancel.clicked.connect(self.reject)
        
        self.btn_ok = QPushButton("确认")
        self.btn_ok.setObjectName("mainBtn")
        self.btn_ok.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_ok.clicked.connect(self.accept)
        
        # Press Enter in input will trigger OK
        self.input_edit.returnPressed.connect(self.accept)
        
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_ok)
        layout.addLayout(btn_layout)

    def toggle_list(self):
        self.list_expanded = not self.list_expanded
        self.list_widget.setVisible(self.list_expanded)
        self.toggle_btn.setText("▲ 收起历史类别列表" if self.list_expanded else "▼ 展开历史类别列表")
        self.adjustSize()

    def on_item_clicked(self, item):
        self.input_edit.setText(item.text())

    def get_text(self):
        return self.input_edit.text().strip()
