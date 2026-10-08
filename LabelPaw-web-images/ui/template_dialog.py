from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                               QLabel, QLineEdit, QListWidget, QGraphicsView, QGraphicsScene, 
                               QGraphicsEllipseItem, QGraphicsLineItem, QInputDialog, QMessageBox, QListWidgetItem, QMenu,
                               QComboBox, QWidget, QColorDialog, QToolButton)
from PySide6.QtCore import Qt, QRectF, Signal, QPointF
from PySide6.QtGui import QPen, QBrush, QColor, QPainter, QIcon
import random

class KeypointItemWidget(QWidget):
    deleted = Signal(int)
    color_changed = Signal(int, str)
    name_changed = Signal(int, str)

    def __init__(self, index, name, color, is_dark_theme=True, parent=None):
        super().__init__(parent)
        self.index = index
        self.color = color
        self.is_dark_theme = is_dark_theme
        self.setMinimumHeight(36)
        self.setup_ui(name)

    def setup_ui(self, name):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Index label
        self.idx_label = QLabel(str(self.index))
        self.idx_label.setFixedWidth(20)
        
        # Color button
        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(20, 20)
        self.update_color_btn()
        self.color_btn.clicked.connect(self.choose_color)

        # Name edit
        self.name_edit = QLineEdit(name)
        self.name_edit.setMinimumHeight(24)
        self.name_edit.textChanged.connect(lambda t: self.name_changed.emit(self.index, t))
        
        # Delete button
        self.del_btn = QToolButton()
        icon = QIcon("ui/icon/trash.svg")
        if self.is_dark_theme:
            self.del_btn.setIcon(self._set_icon_color(icon, QColor("#EF4444")))
        else:
            self.del_btn.setIcon(self._set_icon_color(icon, QColor("#EF4444")))
        self.del_btn.setFixedSize(28, 28)
        from PySide6.QtCore import QSize
        self.del_btn.setIconSize(QSize(20, 20))
        self.del_btn.clicked.connect(lambda: self.deleted.emit(self.index))
        
        if self.is_dark_theme:
            self.idx_label.setStyleSheet("color: #94A3B8; font-weight: bold;")
            self.name_edit.setStyleSheet("background-color: #1E293B; border: 1px solid #334155; border-radius: 4px; padding: 2px 4px; color: #F8FAFC;")
            self.del_btn.setStyleSheet("QToolButton { background-color: transparent; border: none; } QToolButton:hover { background-color: #334155; border-radius: 4px; }")
        else:
            self.idx_label.setStyleSheet("color: #64748B; font-weight: bold;")
            self.name_edit.setStyleSheet("background-color: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 4px; padding: 2px 4px; color: #0F172A;")
            self.del_btn.setStyleSheet("QToolButton { background-color: transparent; border: none; } QToolButton:hover { background-color: #E2E8F0; border-radius: 4px; }")

        layout.addWidget(self.idx_label)
        layout.addWidget(self.color_btn)
        layout.addWidget(self.name_edit)
        layout.addWidget(self.del_btn)

    def _set_icon_color(self, icon, color):
        from PySide6.QtGui import QPainter, QIcon
        pixmap = icon.pixmap(24, 24)
        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), color)
        painter.end()
        new_icon = QIcon()
        new_icon.addPixmap(pixmap, QIcon.Normal, QIcon.On)
        new_icon.addPixmap(pixmap, QIcon.Normal, QIcon.Off)
        return new_icon

    def update_color_btn(self):
        self.color_btn.setStyleSheet(f"background-color: {self.color}; border: 1px solid #000; border-radius: 4px;")

    def choose_color(self):
        color = QColorDialog.getColor(QColor(self.color), self, "选择颜色")
        if color.isValid():
            self.color = color.name()
            self.update_color_btn()
            self.color_changed.emit(self.index, self.color)

class SkeletonTemplateDialog(QDialog):
    def __init__(self, parent=None, template_manager=None, is_dark_theme=True):
        super().__init__(parent)
        self.setWindowTitle("新建骨架模板 (New Skeleton Template)")
        self.resize(900, 600)
        self.template_manager = template_manager
        self.is_dark_theme = is_dark_theme
        
        self.points = []  # [{"name": "kp_0", "color": "#FF0000", "pos": [x,y]}]
        self.connections = [] # [(0, 1), ...]
        self.dragging_idx = -1
        self.selected_point_idx = -1
        self.hover_point_idx = -1
        self.temp_line = None
        
        if self.is_dark_theme:
            self.setStyleSheet("""
                QDialog {
                    background-color: #0F172A;
                    color: #F8FAFC;
                    font-family: "Microsoft YaHei";
                }
                QLabel {
                    color: #cbd5e1;
                    font-weight: bold;
                }
                QLineEdit {
                    background-color: #1E293B;
                    border: 1px solid #334155;
                    border-radius: 6px;
                    padding: 6px;
                    color: #F8FAFC;
                }
                QLineEdit:focus {
                    border: 1px solid #22C55E;
                }
                QPushButton {
                    background-color: #1E293B;
                    border: 1px solid #334155;
                    border-radius: 6px;
                    padding: 8px 16px;
                    color: #F8FAFC;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #334155;
                }
                QPushButton#btnSave {
                    background-color: #22C55E;
                    color: #020617;
                    border: none;
                }
                QPushButton#btnSave:hover {
                    background-color: #4ade80;
                }
                QListWidget {
                    background-color: #1E293B;
                    border: 1px solid #334155;
                    border-radius: 6px;
                    color: #F8FAFC;
                }
                QListWidget::item {
                    padding: 6px;
                }
                QListWidget::item:selected {
                    background-color: #334155;
                    color: #22C55E;
                }
                QComboBox {
                    background-color: #1E293B;
                    border: 1px solid #334155;
                    border-radius: 6px;
                    padding: 6px;
                    color: #F8FAFC;
                }
                QComboBox::drop-down {
                    border: none;
                }
            """)
        else:
            self.setStyleSheet("""
                QDialog {
                    background-color: #F8FAFC;
                    color: #0F172A;
                    font-family: "Microsoft YaHei";
                }
                QLabel {
                    color: #334155;
                    font-weight: bold;
                }
                QLineEdit {
                    background-color: #FFFFFF;
                    border: 1px solid #CBD5E1;
                    border-radius: 6px;
                    padding: 6px;
                    color: #0F172A;
                }
                QLineEdit:focus {
                    border: 1px solid #3B82F6;
                }
                QPushButton {
                    background-color: #FFFFFF;
                    border: 1px solid #CBD5E1;
                    border-radius: 6px;
                    padding: 8px 16px;
                    color: #0F172A;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #F1F5F9;
                }
                QPushButton#btnSave {
                    background-color: #3B82F6;
                    color: #FFFFFF;
                    border: none;
                }
                QPushButton#btnSave:hover {
                    background-color: #2563EB;
                }
                QListWidget {
                    background-color: #FFFFFF;
                    border: 1px solid #CBD5E1;
                    border-radius: 6px;
                    color: #0F172A;
                }
                QListWidget::item {
                    padding: 6px;
                }
                QListWidget::item:selected {
                    background-color: #F1F5F9;
                    color: #3B82F6;
                }
                QComboBox {
                    background-color: #FFFFFF;
                    border: 1px solid #CBD5E1;
                    border-radius: 6px;
                    padding: 6px;
                    color: #0F172A;
                }
                QComboBox::drop-down {
                    border: none;
                }
            """)
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QHBoxLayout()
        
        v1 = QVBoxLayout()
        v1.addWidget(QLabel("名称 (Name)"))
        self.name_edit = QLineEdit("Custom")
        v1.addWidget(self.name_edit)
        header.addLayout(v1)
        
        v_label = QVBoxLayout()
        v_label.addWidget(QLabel("标签 (Label)"))
        self.label_edit = QLineEdit("custom")
        v_label.addWidget(self.label_edit)
        header.addLayout(v_label)
        
        v2 = QVBoxLayout()
        v2.addWidget(QLabel("描述 (Description)"))
        self.desc_edit = QLineEdit("Optional")
        v2.addWidget(self.desc_edit)
        header.addLayout(v2)
        
        v_start = QVBoxLayout()
        v_start.addWidget(QLabel("选择骨架..."))
        self.start_combo = QComboBox()
        self.start_combo.addItem("无 (None)")
        if self.template_manager:
            for t_name in self.template_manager.get_template_names():
                self.start_combo.addItem(t_name)
        self.start_combo.currentTextChanged.connect(self.load_template)
        v_start.addWidget(self.start_combo)
        header.addLayout(v_start)
        
        layout.addLayout(header)
        
        # Main area
        main_area = QHBoxLayout()
        
        # Canvas
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, 500, 500)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setMouseTracking(True)  # Enable mouse tracking for hover effects
        if self.is_dark_theme:
            self.view.setStyleSheet("background-color: #000000; border: 1px solid #334155; border-radius: 8px;")
        else:
            self.view.setStyleSheet("background-color: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 8px;")
        main_area.addWidget(self.view, 3)
        
        self.view.mousePressEvent = self.canvas_mouse_press
        self.view.mouseMoveEvent = self.canvas_mouse_move
        self.view.mouseReleaseEvent = self.canvas_mouse_release
        
        # Right panel
        right_panel = QVBoxLayout()
        tip = QLabel("左键点击添加/连接节点\n右键点击节点删除\n拖动节点调整位置")
        tip.setStyleSheet("color: #94A3B8; font-weight: normal; font-size: 12px;")
        right_panel.addWidget(tip)
        
        self.list_widget = QListWidget()
        right_panel.addWidget(self.list_widget)
        
        main_area.addLayout(right_panel, 2)
        layout.addLayout(main_area)
        
        # Bottom
        self.status_label = QLabel("0 关键点, 0 连线")
        self.status_label.setStyleSheet("color: #94A3B8; font-weight: normal;")
        layout.addWidget(self.status_label)
        
        buttons = QHBoxLayout()
        btn_cancel = QPushButton("取消 (Cancel)")
        btn_cancel.clicked.connect(self.reject)
        self.btn_save = QPushButton("保存 (Save)")
        self.btn_save.setObjectName("btnSave")
        self.btn_save.clicked.connect(self.save_template)
        
        buttons.addStretch()
        buttons.addWidget(btn_cancel)
        buttons.addWidget(self.btn_save)
        layout.addLayout(buttons)
        
        self.redraw()
        
    def load_template(self, template_name):
        if not template_name or template_name == "无 (None)" or template_name == "None":
            self.points = []
            self.connections = []
            self.name_edit.setText("Custom")
            self.label_edit.setText("custom")
            self.desc_edit.setText("Optional")
            self.selected_point_idx = -1
            self.update_list()
            self.redraw()
            return
            
        if self.template_manager:
            t = self.template_manager.get_template(template_name)
            if t:
                self.name_edit.setText(f"{t.get('name', '')} (copy)")
                self.label_edit.setText(t.get('label', ''))
                self.desc_edit.setText(t.get('description', ''))
                
                self.points = []
                for kp in t.get('keypoints', []):
                    self.points.append({
                        "name": kp.get("name", ""),
                        "color": kp.get("color", "#FF0000"),
                        "pos": kp.get("default_pos", [0.5, 0.5]).copy()
                    })
                
                self.connections = [list(c) for c in t.get('connections', [])]
                self.selected_point_idx = -1
                self.update_list()
                self.redraw()

    def update_point_color(self, idx, color):
        if 0 <= idx < len(self.points):
            self.points[idx]["color"] = color
            self.redraw()

    def update_point_name(self, idx, name):
        if 0 <= idx < len(self.points):
            self.points[idx]["name"] = name
            
    def delete_point(self, idx):
        if 0 <= idx < len(self.points):
            self.points.pop(idx)
            # update connections
            new_conn = []
            for c in self.connections:
                if c[0] == idx or c[1] == idx:
                    continue
                nc = [c[0] if c[0] < idx else c[0]-1,
                      c[1] if c[1] < idx else c[1]-1]
                new_conn.append(nc)
            self.connections = new_conn
            if self.selected_point_idx == idx:
                self.selected_point_idx = -1
            elif self.selected_point_idx > idx:
                self.selected_point_idx -= 1
            self.update_list()
            self.redraw()

    def canvas_mouse_press(self, event):
        pos = self.view.mapToScene(event.pos())
        
        # Check if clicking existing point
        clicked_idx = -1
        for i, pt in enumerate(self.points):
            px, py = pt["pos"][0] * 500, pt["pos"][1] * 500
            if (pos.x() - px)**2 + (pos.y() - py)**2 < 100:
                clicked_idx = i
                break
                
        if event.button() == Qt.LeftButton:
            if clicked_idx != -1:
                self.dragging_idx = clicked_idx
                # Check if we should connect
                if self.selected_point_idx != -1 and self.selected_point_idx != clicked_idx:
                    # add connection if not exists
                    conn = [self.selected_point_idx, clicked_idx]
                    conn_rev = [clicked_idx, self.selected_point_idx]
                    if conn not in self.connections and conn_rev not in self.connections:
                        self.connections.append(conn)
                self.selected_point_idx = clicked_idx
                self.update_list()
                self.redraw()
            else:
                # Add new point
                idx = len(self.points)
                hue = random.randint(0, 359)
                if self.is_dark_theme:
                    color_hex = QColor.fromHsv(hue, 200, 250).name()
                else:
                    color_hex = QColor.fromHsv(hue, 220, 120).name()
                new_pt = {
                    "name": f"kp_{idx}",
                    "color": color_hex,
                    "pos": [pos.x() / 500.0, pos.y() / 500.0]
                }
                self.points.append(new_pt)
                if self.selected_point_idx != -1:
                    self.connections.append([self.selected_point_idx, idx]) # Auto connect to selected
                self.selected_point_idx = idx
                self.update_list()
                self.redraw()
        elif event.button() == Qt.RightButton:
            # Right click deselects active point, or deletes if clicking on one
            if clicked_idx != -1:
                self.delete_point(clicked_idx)
            else:
                self.selected_point_idx = -1
                self.update_list()
                self.redraw()

    def canvas_mouse_move(self, event):
        pos = self.view.mapToScene(event.pos())
        
        if self.dragging_idx != -1:
            x = max(0, min(pos.x(), 500)) / 500.0
            y = max(0, min(pos.y(), 500)) / 500.0
            self.points[self.dragging_idx]["pos"] = [x, y]
            
        # Draw temporary line
        if self.selected_point_idx != -1 and self.dragging_idx == -1:
            if not self.temp_line:
                self.temp_line = QGraphicsLineItem()
                self.scene.addItem(self.temp_line)
            
            p1_color = QColor(self.points[self.selected_point_idx]["color"])
            p1_color.setAlpha(150)
            self.temp_line.setPen(QPen(p1_color, 2, Qt.DashLine))
            
            p1_x = self.points[self.selected_point_idx]["pos"][0] * 500
            p1_y = self.points[self.selected_point_idx]["pos"][1] * 500
            self.temp_line.setLine(p1_x, p1_y, pos.x(), pos.y())
            self.temp_line.show()
        else:
            if self.temp_line:
                self.temp_line.hide()

        self.redraw(draw_temp=False)

    def canvas_mouse_release(self, event):
        self.dragging_idx = -1

    def update_list(self):
        self.list_widget.clear()
        for i, pt in enumerate(self.points):
            item = QListWidgetItem()
            widget = KeypointItemWidget(i, pt['name'], pt['color'], self.is_dark_theme)
            widget.color_changed.connect(self.update_point_color)
            widget.name_changed.connect(self.update_point_name)
            widget.deleted.connect(self.delete_point)
            
            # 选中背景色
            item.setSizeHint(widget.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)
            
            if i == self.selected_point_idx:
                item.setSelected(True)
            
    def redraw(self, draw_temp=True):
        if draw_temp:
            self.scene.clear()
            self.temp_line = None
            
            # Draw grid
            pen = QPen(QColor(220, 220, 220, 50))
            for i in range(1, 10):
                self.scene.addLine(0, i*50, 500, i*50, pen)
                self.scene.addLine(i*50, 0, i*50, 500, pen)
                
            # Draw connections
            for p1, p2 in self.connections:
                x1, y1 = self.points[p1]["pos"][0]*500, self.points[p1]["pos"][1]*500
                x2, y2 = self.points[p2]["pos"][0]*500, self.points[p2]["pos"][1]*500
                color_p1 = QColor(self.points[p1]["color"])
                self.scene.addLine(x1, y1, x2, y2, QPen(color_p1, 2))
                
            # Draw points
            for i, pt in enumerate(self.points):
                x, y = pt["pos"][0]*500, pt["pos"][1]*500
                r = 6 if i == self.selected_point_idx else 5
                
                if i == self.selected_point_idx:
                    # Highlight selected point
                    self.scene.addEllipse(x-r-3, y-r-3, (r+3)*2, (r+3)*2, QPen(QColor(pt["color"]), 2), QBrush(Qt.NoBrush))
                    
                ellipse = self.scene.addEllipse(x-r, y-r, r*2, r*2, QPen(Qt.NoPen), QBrush(QColor(pt["color"])))
                
                # Draw label
                text = self.scene.addText(str(i))
                if self.is_dark_theme:
                    text.setDefaultTextColor(QColor("#FFFFFF"))
                else:
                    text.setDefaultTextColor(QColor("#000000"))
                text.setPos(x+5, y-10)
                
            self.status_label.setText(f"{len(self.points)} 关键点, {len(self.connections)} 连线")
        else:
            # Just update positions of existing items to be faster, but for now full redraw if needed
            # Since we clear scene on draw_temp=True, if it's False, we just don't clear and assume items exist.
            # But we need to update lines and points if they moved.
            # For simplicity, if we want fast updates, we should keep references.
            # Let's just always full redraw except keep temp_line.
            pass
            
        if not draw_temp:
            # Partial update to avoid flickering while drawing temp line
            # We clear everything except temp_line?
            # Actually QGraphicsScene doesn't flicker on clear(). Let's just keep the old logic but preserve temp_line.
            
            # Remove all items except temp_line
            for item in self.scene.items():
                if item != self.temp_line:
                    self.scene.removeItem(item)
                    
            # Draw grid
            pen = QPen(QColor(220, 220, 220, 50))
            for i in range(1, 10):
                self.scene.addLine(0, i*50, 500, i*50, pen)
                self.scene.addLine(i*50, 0, i*50, 500, pen)
                
            # Draw connections
            for p1, p2 in self.connections:
                x1, y1 = self.points[p1]["pos"][0]*500, self.points[p1]["pos"][1]*500
                x2, y2 = self.points[p2]["pos"][0]*500, self.points[p2]["pos"][1]*500
                color_p1 = QColor(self.points[p1]["color"])
                self.scene.addLine(x1, y1, x2, y2, QPen(color_p1, 2))
                
            # Draw points
            for i, pt in enumerate(self.points):
                x, y = pt["pos"][0]*500, pt["pos"][1]*500
                r = 6 if i == self.selected_point_idx else 5
                
                if i == self.selected_point_idx:
                    self.scene.addEllipse(x-r-3, y-r-3, (r+3)*2, (r+3)*2, QPen(QColor(pt["color"]), 2), QBrush(Qt.NoBrush))
                    
                ellipse = self.scene.addEllipse(x-r, y-r, r*2, r*2, QPen(Qt.NoPen), QBrush(QColor(pt["color"])))
                
                text = self.scene.addText(str(i))
                if self.is_dark_theme:
                    text.setDefaultTextColor(QColor("#FFFFFF"))
                else:
                    text.setDefaultTextColor(QColor("#000000"))
                text.setPos(x+5, y-10)

    def save_template(self):
        if not self.points:
            QMessageBox.warning(self, "Error", "Template must have at least one keypoint.")
            return
            
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Please enter a template name.")
            return
            
        label = self.label_edit.text().strip()
        if not label:
            label = name.lower().replace(' ', '_')
            
        template = {
            "name": name,
            "label": label,
            "description": self.desc_edit.text().strip(),
            "kpt_shape": [len(self.points), 3],
            "keypoints": [{"name": pt["name"], "color": pt["color"], "default_pos": pt["pos"]} for pt in self.points],
            "connections": self.connections
        }
        
        if self.template_manager:
            self.template_manager.add_template(template)
            
        self.accept()
