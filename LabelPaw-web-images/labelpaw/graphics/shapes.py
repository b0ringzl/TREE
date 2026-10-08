from PySide6.QtWidgets import QGraphicsItem, QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsEllipseItem, \
    QGraphicsTextItem, QGraphicsLineItem, QGraphicsObject, QMenu
from PySide6.QtGui import QPen, QBrush, QColor, QPolygonF, QFont, QTransform, QPainter, QAction, QPainterPath, QPainterPathStroker
from PySide6.QtCore import Qt, QRectF, QPointF
import math


def point_to_segment_dist(p, a, b):
    px, py = p.x(), p.y()
    ax, ay = a.x(), a.y()
    bx, by = b.x(), b.y()

    ab_dist_sq = (bx - ax) ** 2 + (by - ay) ** 2
    if ab_dist_sq == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5, a

    t = max(0, min(1, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / ab_dist_sq))
    proj_x = ax + t * (bx - ax)
    proj_y = ay + t * (by - ay)

    dist = ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5
    return dist, QPointF(proj_x, proj_y)


def clamp_item_position(item, proposed_pos, overflow_ratio=0.0):
    scene = item.scene()
    if not scene: return proposed_pos
    rect = scene.sceneRect()
    shape_rect = item.boundingRect()

    allow_w = shape_rect.width() * overflow_ratio
    allow_h = shape_rect.height() * overflow_ratio

    min_x = rect.left() - shape_rect.left() - allow_w
    max_x = rect.right() - shape_rect.right() + allow_w
    min_y = rect.top() - shape_rect.top() - allow_h
    max_y = rect.bottom() - shape_rect.bottom() + allow_h

    new_x, new_y = proposed_pos.x(), proposed_pos.y()
    if min_x <= max_x:
        new_x = max(min_x, min(new_x, max_x))
    if min_y <= max_y:
        new_y = max(min_y, min(new_y, max_y))

    return QPointF(new_x, new_y)


def get_current_image_height(shape_item=None):
    # 1. Try to get it from the item's own scene
    if shape_item:
        scene = shape_item.scene()
        if scene and hasattr(scene, 'img_item') and scene.img_item:
            pixmap = scene.img_item.pixmap()
            if pixmap and not pixmap.isNull():
                return pixmap.height()
    
    # 2. Try to get it from the active window's current image/scene
    from PySide6.QtWidgets import QApplication
    for widget in QApplication.topLevelWidgets():
        if hasattr(widget, 'scene') and widget.scene:
            scene = widget.scene
            if hasattr(scene, 'img_item') and scene.img_item:
                pixmap = scene.img_item.pixmap()
                if pixmap and not pixmap.isNull():
                    return pixmap.height()
                
    return 720 # Fallback


class BaseShape:
    def setup_style(self, item):
        self.normal_pen = QPen(QColor(28, 126, 214), 2)
        self.normal_brush = QBrush(QColor(28, 126, 214, 50))
        self.hover_brush = QBrush(QColor(28, 126, 214, 120))

        item.setPen(self.normal_pen)
        item.setBrush(self.normal_brush)
        item.setFlags(
            QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        item.setAcceptHoverEvents(True)

    def apply_hover_enter(self, item):
        if not getattr(item, 'is_temp', False):
            scene = item.scene()
            if scene and getattr(scene, 'mode', None) == 2:  # CanvasMode.POLY
                if len(getattr(scene, 'poly_pts', [])) > 0:
                    return
            if not item.isSelected():
                item.setZValue(50.0)
            if hasattr(item, 'setBrush'):
                item.setBrush(self.hover_brush)
            elif hasattr(item, 'rect_item') and hasattr(item.rect_item, 'setBrush'):
                c = getattr(item, '_base_color', None)
                if c:
                    item.rect_item.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), 120)))
                else:
                    item.rect_item.setBrush(self.hover_brush)
            item.setCursor(Qt.PointingHandCursor)
            if scene and hasattr(scene, 'canvas_item_hovered'):
                scene.canvas_item_hovered.emit(item, True)

    def apply_hover_leave(self, item):
        if not getattr(item, 'is_temp', False):
            if not item.isSelected():
                item.setZValue(0.0)
            if hasattr(item, 'setBrush'):
                item.setBrush(self.normal_brush)
            elif hasattr(item, 'rect_item') and hasattr(item.rect_item, 'setBrush'):
                c = getattr(item, '_base_color', None)
                if c:
                    item.rect_item.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), 50)))
                else:
                    item.rect_item.setBrush(self.normal_brush)
            item.setCursor(Qt.ArrowCursor)
            scene = item.scene()
            if scene and hasattr(scene, 'canvas_item_hovered'):
                scene.canvas_item_hovered.emit(item, False)

    def set_color(self, color):
        """dynamically update shape color based on class color"""
        self.normal_pen = QPen(color, 2)
        self.normal_brush = QBrush(QColor(color.red(), color.green(), color.blue(), 50))
        self.hover_brush = QBrush(QColor(color.red(), color.green(), color.blue(), 120))
        self.setPen(self.normal_pen)
        self.setBrush(self.normal_brush)
        if hasattr(self, 'label_text') and self.label_text:
            self.label_text.setDefaultTextColor(color)
        for attr in ['lt_handle', 'rt_handle', 'lb_handle', 'rb_handle', 't_handle', 'b_handle', 'l_handle', 'r_handle']:
            h = getattr(self, attr, None)
            if h:
                h.setPen(QPen(color, 1.5))
        if hasattr(self, 'handles'):
            for h in self.handles:
                if isinstance(h, HandleItem):
                    h.setPen(QPen(color, 1.5))

    def setup_label(self, item):
        self.label_text = QGraphicsTextItem(item)
        self.label_text.setDefaultTextColor(QColor(255, 255, 255))
        self.label_text.setFont(QFont("Arial", 10, QFont.Bold))
        self.label_text.setZValue(1001)

    def update_label_position(self, item):
        if not hasattr(self, 'label_text') or not self.label_text:
            return

        bound_rect = item.boundingRect()
        label_w = self.label_text.boundingRect().width()
        label_h = self.label_text.boundingRect().height()
        
        # Default position: above the bounding box
        x = bound_rect.center().x() - label_w / 2
        y = bound_rect.top() - 22
        
        # Get scene coordinates for this position
        scene_pos = item.mapToScene(QPointF(x, y))
        
        # Get image bounds from scene or active main window
        img_w, img_h = 1280, 720
        scene = item.scene()
        if scene and hasattr(scene, 'img_item') and scene.img_item:
            pix = scene.img_item.pixmap()
            if pix and not pix.isNull():
                img_w = pix.width()
                img_h = pix.height()
        else:
            from PySide6.QtWidgets import QApplication
            for widget in QApplication.topLevelWidgets():
                if hasattr(widget, 'scene') and widget.scene:
                    sc = widget.scene
                    if hasattr(sc, 'img_item') and sc.img_item:
                        pix = sc.img_item.pixmap()
                        if pix and not pix.isNull():
                            img_w = pix.width()
                            img_h = pix.height()
                            break

        # Check if the label goes above the image top, or beyond left/right edges
        margin = 5.0
        if scene_pos.y() < margin or scene_pos.x() < margin or (scene_pos.x() + label_w) > (img_w - margin):
            # Center the label inside the bounding box
            x = bound_rect.center().x() - label_w / 2
            y = bound_rect.center().y() - label_h / 2
            
        self.label_text.setPos(x, y)

    def update_label_text(self, text):
        if hasattr(self, 'label_text') and self.label_text:
            self.label_text.setPlainText(text)
            self.label_text.show()

    def update_label_visibility(self, item, is_selected=False, is_hovered=False):
        if hasattr(self, 'label_text') and self.label_text:
            self.label_text.show()


class HandleItem(QGraphicsItem):
    def __init__(self, parent, cursor_type=Qt.SizeAllCursor, handle_type='circle'):
        super().__init__(parent)
        self.cursor_type = cursor_type
        self.handle_type = handle_type  # 'circle', 'h_capsule', 'v_capsule'
        self.setAcceptHoverEvents(True)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setZValue(1000)
        self.hide()
        self._mouse_press_pos = None
        self._is_moved = False
        
        # Dimensions
        if self.handle_type == 'h_capsule':
            self.w, self.h = 16, 6
        elif self.handle_type == 'v_capsule':
            self.w, self.h = 6, 16
        else:  # 'circle'
            self.w, self.h = 7, 7
            
        self._pen_color = QColor(28, 126, 214)

    def boundingRect(self):
        return QRectF(-self.w / 2, -self.h / 2, self.w, self.h)

    def setPen(self, pen):
        self._pen_color = pen.color()
        self.update()

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(self._pen_color, 1.5))
        
        if self.handle_type == 'circle':
            painter.drawEllipse(self.boundingRect())
        else:
            r = min(self.w, self.h) / 2
            painter.drawRoundedRect(self.boundingRect(), r, r)

    def hoverEnterEvent(self, event):
        self.setCursor(self.cursor_type)
        if hasattr(self.parentItem(), '_hovered'):
            self.parentItem()._hovered = True
            self.parentItem()._update_handle_visibility()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        self._mouse_press_pos = event.pos()
        self._is_moved = False
        if self.parentItem():
            self.parentItem().setSelected(True)
            self.parentItem().setFlag(QGraphicsItem.ItemIsMovable, False)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._mouse_press_pos and (event.pos() - self._mouse_press_pos).manhattanLength() > 2:
            self._is_moved = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.parentItem():
            self.parentItem().setFlag(QGraphicsItem.ItemIsMovable, True)

        if not self._is_moved and event.button() == Qt.LeftButton:
            parent = self.parentItem()
            if hasattr(parent, 'remove_handle'):
                parent.remove_handle(self)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene() and self.parentItem():
            parent = self.parentItem()
            scene_pos = parent.mapToScene(value)
            rect = self.scene().sceneRect()
            clamped_x = max(rect.left(), min(scene_pos.x(), rect.right()))
            clamped_y = max(rect.top(), min(scene_pos.y(), rect.bottom()))
            return super().itemChange(change, parent.mapFromScene(QPointF(clamped_x, clamped_y)))

        if change == QGraphicsItem.ItemPositionHasChanged and self.scene():
            parent = self.parentItem()
            if hasattr(parent, 'update_from_handle') and not getattr(parent, '_updating_handles', False):
                parent.update_from_handle(self)
            elif hasattr(parent, 'update_from_handles') and not getattr(parent, '_updating_handles', False):
                parent.update_from_handles()
        return super().itemChange(change, value)


class RectShape(QGraphicsRectItem, BaseShape):
    def __init__(self, rect, label=""):
        super().__init__(rect)
        self.setup_style(self)
        self.label = label
        self._updating_handles = False
        self._hovered = False

        self.setup_label(self)
        if label:
            self.update_label_text(label)
            self.update_label_position(self)

        # 4 corners - white circles
        self.lt_handle = HandleItem(self, Qt.SizeFDiagCursor, 'circle')
        self.rt_handle = HandleItem(self, Qt.SizeBDiagCursor, 'circle')
        self.lb_handle = HandleItem(self, Qt.SizeBDiagCursor, 'circle')
        self.rb_handle = HandleItem(self, Qt.SizeFDiagCursor, 'circle')
        
        # 4 edge borders - pill capsule style!
        self.t_handle = HandleItem(self, Qt.SizeVerCursor, 'h_capsule')
        self.b_handle = HandleItem(self, Qt.SizeVerCursor, 'h_capsule')
        self.l_handle = HandleItem(self, Qt.SizeHorCursor, 'v_capsule')
        self.r_handle = HandleItem(self, Qt.SizeHorCursor, 'v_capsule')
        
        self.update_handles_pos()

    def hoverEnterEvent(self, event):
        self.apply_hover_enter(self)
        self._hovered = True
        self._update_handle_visibility()
        self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.apply_hover_leave(self)
        self._hovered = False
        self._update_handle_visibility()
        self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=False)
        super().hoverLeaveEvent(event)

    def _update_handle_visibility(self):
        scene = self.scene()
        is_drawing_poly = False
        if scene and getattr(scene, 'mode', None) == 2:
            if len(getattr(scene, 'poly_pts', [])) > 0:
                is_drawing_poly = True
                
        visible = self.isSelected() or (self._hovered and not is_drawing_poly)
        for h in [self.lt_handle, self.rt_handle, self.lb_handle, self.rb_handle,
                  self.t_handle, self.b_handle, self.l_handle, self.r_handle]:
            h.setVisible(visible)

    def update_handles_pos(self):
        self._updating_handles = True
        r = self.rect()
        # Corners
        self.lt_handle.setPos(r.topLeft())
        self.rt_handle.setPos(r.topRight())
        self.lb_handle.setPos(r.bottomLeft())
        self.rb_handle.setPos(r.bottomRight())
        # Edges
        self.t_handle.setPos(r.left() + r.width() / 2, r.top())
        self.b_handle.setPos(r.left() + r.width() / 2, r.bottom())
        self.l_handle.setPos(r.left(), r.top() + r.height() / 2)
        self.r_handle.setPos(r.right(), r.top() + r.height() / 2)
        self._updating_handles = False

    def update_from_handle(self, dragged_handle):
        if self._updating_handles:
            return

        self._updating_handles = True
        hx, hy = dragged_handle.pos().x(), dragged_handle.pos().y()

        if dragged_handle == self.t_handle:
            self.lt_handle.setPos(self.lt_handle.pos().x(), hy)
            self.rt_handle.setPos(self.rt_handle.pos().x(), hy)
        elif dragged_handle == self.b_handle:
            self.lb_handle.setPos(self.lb_handle.pos().x(), hy)
            self.rb_handle.setPos(self.rb_handle.pos().x(), hy)
        elif dragged_handle == self.l_handle:
            self.lt_handle.setPos(hx, self.lt_handle.pos().y())
            self.lb_handle.setPos(hx, self.lb_handle.pos().y())
        elif dragged_handle == self.r_handle:
            self.rt_handle.setPos(hx, self.rt_handle.pos().y())
            self.rb_handle.setPos(hx, self.rb_handle.pos().y())
        elif dragged_handle == self.lt_handle:
            self.rt_handle.setPos(self.rt_handle.pos().x(), hy)
            self.lb_handle.setPos(hx, self.lb_handle.pos().y())
        elif dragged_handle == self.rt_handle:
            self.lt_handle.setPos(self.lt_handle.pos().x(), hy)
            self.rb_handle.setPos(hx, self.rb_handle.pos().y())
        elif dragged_handle == self.lb_handle:
            self.rb_handle.setPos(self.rb_handle.pos().x(), hy)
            self.lt_handle.setPos(hx, self.lt_handle.pos().y())
        elif dragged_handle == self.rb_handle:
            self.lb_handle.setPos(self.lb_handle.pos().x(), hy)
            self.rt_handle.setPos(hx, self.rt_handle.pos().y())

        min_x = min(self.lt_handle.pos().x(), self.rb_handle.pos().x())
        max_x = max(self.lt_handle.pos().x(), self.rb_handle.pos().x())
        min_y = min(self.lt_handle.pos().y(), self.rb_handle.pos().y())
        max_y = max(self.lt_handle.pos().y(), self.rb_handle.pos().y())

        self.setRect(QRectF(min_x, min_y, max_x - min_x, max_y - min_y))
        self.update_label_position(self)
        self.update_handles_pos()
        self._updating_handles = False

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and not getattr(self, 'is_temp', False):
            return super().itemChange(change, clamp_item_position(self, value))

        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.setZValue(100.0 if self.isSelected() else 0.0)
            self._update_handle_visibility()
            self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=self._hovered)
        elif change == QGraphicsItem.ItemPositionHasChanged:
            self.update_label_position(self)
        return super().itemChange(change, value)


from labelpaw.config.translations import KEYPOINT_TRANSLATIONS

class KeypointHandle(QGraphicsEllipseItem):
    def __init__(self, index, kp_info, parent):
        r = 4
        super().__init__(-r, -r, r * 2, r * 2, parent)
        self.index = index
        self.kp_name = kp_info.get("name", str(index))
        color_hex = kp_info.get("color", "#00FF00")
        self.color = QColor(color_hex)
        
        self.visible_state = 2  # 2: visible, 1: occluded, 0: hidden
        
        self.setBrush(QBrush(self.color))
        self.setPen(QPen(QColor(255, 255, 255), 1.5))
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setZValue(1002)
        
        # Tooltip for hover
        display_name = KEYPOINT_TRANSLATIONS.get(self.kp_name, self.kp_name)
        if display_name == self.kp_name and self.kp_name.startswith("pt_"):
            display_name = self.kp_name.replace("pt_", "点_")
            
        self.setToolTip(f"{self.index}: {display_name}")
        
    def hoverEnterEvent(self, event):
        parent = self.parentItem()
        is_editing = parent and parent.isSelected()

        if not is_editing:
            self.setCursor(Qt.PointingHandCursor)
            if parent and hasattr(parent, 'set_hover_state'):
                parent.set_hover_state(True)
            super().hoverEnterEvent(event)
            return

        from PySide6.QtGui import QCursor, QIcon
        icon_path = "ui/icon/hand-grabbing-duotone.svg"
        
        try:
            pixmap = QIcon(icon_path).pixmap(24, 24)
            self.setCursor(QCursor(pixmap, 12, 12))
        except:
            self.setCursor(Qt.OpenHandCursor)
            
        # highlight
        self.setPen(QPen(QColor(255, 255, 0), 2))
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        self.setSelected(False)
        parent = self.parentItem()
        is_editing = parent and parent.isSelected()

        if not is_editing:
            self.unsetCursor()
            if parent and hasattr(parent, 'set_hover_state'):
                parent.set_hover_state(False)
            super().hoverLeaveEvent(event)
            return

        self.unsetCursor()
        self.setPen(QPen(QColor(255, 255, 255), 1.5))
        super().hoverLeaveEvent(event)
        
    def mousePressEvent(self, event):
        parent = self.parentItem()
        is_editing = parent and parent.isSelected()

        if not is_editing:
            event.ignore()
            return

        if event.button() == Qt.RightButton:
            # Context menu for visibility
            menu = QMenu()
            # 应用暗色/亮色现代样式
            menu.setStyleSheet("""
                QMenu {
                    background-color: #2b2b2b;
                    color: #e0e0e0;
                    border: 1px solid #444;
                    border-radius: 6px;
                    padding: 4px;
                    font-family: "Segoe UI", "Microsoft YaHei";
                    font-size: 13px;
                }
                QMenu::item {
                    padding: 6px 24px 6px 12px;
                    border-radius: 4px;
                }
                QMenu::item:selected {
                    background-color: #3a3a3a;
                    color: #ffffff;
                }
            """)
            act_vis = menu.addAction("可见 (Visible) [2]")
            act_occ = menu.addAction("遮挡 (Occluded) [1]")
            act_hid = menu.addAction("隐藏 (Hidden) [0]")
            
            action = menu.exec(event.screenPos())
            if action == act_vis:
                self.set_visibility(2)
            elif action == act_occ:
                self.set_visibility(1)
            elif action == act_hid:
                self.set_visibility(0)
            event.accept()
        elif event.button() == Qt.LeftButton:
            self.setCursor(Qt.ClosedHandCursor)
            # 锁定父级位置，确保只移动当前关键点
            if self.parentItem():
                self.parentItem().setFlag(QGraphicsItem.ItemIsMovable, False)
                self.parentItem().setSelected(True)
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            from PySide6.QtGui import QCursor, QIcon
            parent = self.parentItem()
            is_editing = parent and parent.isSelected()
            icon_path = "ui/icon/hand-grabbing-duotone.svg" if is_editing else "ui/icon/hand-grabbing-duotone.svg"
            try:
                pixmap = QIcon(icon_path).pixmap(24, 24)
                self.setCursor(QCursor(pixmap, 12, 12))
            except:
                self.setCursor(Qt.OpenHandCursor)
            # 恢复父级可移动状态
            self.parentItem().setFlag(QGraphicsItem.ItemIsMovable, True)
            
            # 通知主场景更新状态（保存撤销记录）
            # 必须检查当前对象是否处于非预览状态
            if not getattr(self.parentItem(), 'is_temp', False) and self.scene() and hasattr(self.scene(), 'state_changed'):
                self.scene().state_changed.emit()
                
        super().mouseReleaseEvent(event)
            
    def set_visibility(self, state):
        self.visible_state = state
        if state == 2:
            self.setBrush(QBrush(self.color))
            self.setOpacity(1.0)
        elif state == 1:
            self.setBrush(QBrush(QColor(150, 150, 150)))
            self.setOpacity(0.8)
        elif state == 0:
            self.setBrush(QBrush(Qt.NoBrush))
            self.setOpacity(0.3)
        if hasattr(self.parentItem(), 'update_lines'):
            self.parentItem().update_lines()
            
    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            if hasattr(self.parentItem(), 'update_lines'):
                self.parentItem().update_lines()
            if hasattr(self.parentItem(), 'update_bounding_box'):
                self.parentItem().update_bounding_box()
        return super().itemChange(change, value)


class PoseShape(QGraphicsObject, BaseShape):
    def __init__(self, rect, template, label="", is_temp=False):
        super().__init__()
        self.is_temp = is_temp
        self.label = label
        self.template = template
        
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        
        self._is_resizing = False
        self._dragging_handle = None
        self._hovered = False

        # 依赖宽高
        self.box_w = rect.width()
        self.box_h = rect.height()
        # 居中显示
        self.setPos(rect.center())
        
        # Base BBox (can be hidden or shown, usually drawn as dashed)
        self.rect_item = QGraphicsRectItem(self)
        if is_temp:
            # 透明无边框的幽灵预览
            self.rect_item.setPen(QPen(Qt.NoPen))
            self.rect_item.setBrush(QBrush(Qt.NoBrush))
        else:
            self.rect_item.setPen(QPen(Qt.NoPen)) # 默认无边框
            self.rect_item.setBrush(QBrush(Qt.NoBrush))
        
        self.rotate_line = QGraphicsLineItem(self)
        if is_temp:
            self.rotate_line.setPen(QPen(Qt.NoPen))
        else:
            self.rotate_line.setPen(QPen(QColor(50, 255, 50, 200), 1.5))

        self.h_top = OBBHandle('top', self)
        self.h_bottom = OBBHandle('bottom', self)
        self.h_left = OBBHandle('left', self)
        self.h_right = OBBHandle('right', self)
        self.h_rotate = OBBHandle('rotate', self)
        self.h_tl = OBBHandle('tl', self)
        self.h_tr = OBBHandle('tr', self)
        self.h_bl = OBBHandle('bl', self)
        self.h_br = OBBHandle('br', self)

        self.handles = [self.h_top, self.h_bottom, self.h_left, self.h_right, self.h_rotate, self.h_tl, self.h_tr, self.h_bl, self.h_br, self.rotate_line]

        self.kps = []
        self.lines = []
        
        self._is_initializing = True

        # Initialize keypoints
        kps_info = template.get("keypoints", [])
        for i, kp_info in enumerate(kps_info):
            handle = KeypointHandle(i, kp_info, self)
            def_pos = kp_info.get("default_pos", [0.5, 0.5])
            # position relative to bounding box center (-w/2 to w/2)
            px = (def_pos[0] - 0.5) * self.box_w
            py = (def_pos[1] - 0.5) * self.box_h
            handle.setPos(px, py)
            self.kps.append(handle)
            
        # Initialize lines
        connections = template.get("connections", [])
        for p1, p2 in connections:
            if 0 <= p1 < len(self.kps) and 0 <= p2 < len(self.kps):
                line = QGraphicsLineItem(self)
                line.setZValue(1001)
                self.lines.append((line, p1, p2))
                
        self._is_initializing = False
        
        self.update_bounding_box()  # Automatically snap to Ultralytics tight bounds on creation
        self.update_geometry()
        self.update_lines()
        self._update_handle_visibility()
        
        if not is_temp:
            self.setup_label(self)
            if label: self.update_label_text(label)
            self.update_label_position(self)
            
        if is_temp:
            for h in self.handles:
                h.hide()

    def set_color(self, color):
        """PoseShape uses QGraphicsObject, not QGraphicsRectItem, so override set_color"""
        if hasattr(self, 'label_text') and self.label_text:
            self.label_text.setDefaultTextColor(color)

    def update_geometry(self):
        w, h = self.box_w, self.box_h
        self.rect_item.setRect(-w / 2, -h / 2, w, h)

        self.h_top.setPos(0, -h / 2)
        self.h_bottom.setPos(0, h / 2)
        self.h_left.setPos(-w / 2, 0)
        self.h_right.setPos(w / 2, 0)

        self.h_tl.setPos(-w / 2, -h / 2)
        self.h_tr.setPos(w / 2, -h / 2)
        self.h_bl.setPos(-w / 2, h / 2)
        self.h_br.setPos(w / 2, h / 2)

        self.h_rotate.setPos(0, -h / 2 - 30)
        self.rotate_line.setLine(0, -h / 2, 0, -h / 2 - 30)

        self.prepareGeometryChange()
        if not self.is_temp:
            self.update_label_position(self)

    def boundingRect(self):
        r = self.box_h / 2 + 35
        return QRectF(-self.box_w / 2 - 10, -r, self.box_w + 20, r + self.box_h / 2 + 10)

    def shape(self):
        path = QPainterPath()
        
        # Add keypoints to path
        for kp in self.kps:
            if kp.visible_state > 0:
                r = 8
                path.addEllipse(kp.pos(), r, r)
                
        # Add lines to path
        line_path = QPainterPath()
        has_lines = False
        for line_item, p1, p2 in self.lines:
            kp1 = self.kps[p1]
            kp2 = self.kps[p2]
            if kp1.visible_state > 0 and kp2.visible_state > 0:
                line_path.moveTo(kp1.pos())
                line_path.lineTo(kp2.pos())
                has_lines = True
                
        if has_lines:
            stroker = QPainterPathStroker()
            stroker.setWidth(10) # 10 pixels wide hit area for lines
            stroker.setCapStyle(Qt.RoundCap)
            stroker.setJoinStyle(Qt.RoundJoin)
            path.addPath(stroker.createStroke(line_path))
            
        # If the item is selected, we might want the bounding box to be part of the shape
        # But for stricter hit detection matching the user's request, we only return the skeleton.
        # Wait, if we return only the skeleton, then we can only drag the skeleton by its bones/joints.
        # This is exactly what the user wants: "需要靠近关键点或者点与点连接线、在线上才出现小手状态"
        
        # However, to make sure the handles (like rotate/scale) can be clicked, we don't need to add them to `shape()`.
        # Child items (handles) handle their own hit testing.
        return path

    def paint(self, painter, option, widget=None):
        pass
        
    def update_lines(self):
        for line_item, p1, p2 in self.lines:
            kp1 = self.kps[p1]
            kp2 = self.kps[p2]
            line_item.setLine(kp1.pos().x(), kp1.pos().y(), kp2.pos().x(), kp2.pos().y())
            # Color logic based on visibility
            if kp1.visible_state == 0 or kp2.visible_state == 0:
                line_item.hide()
            else:
                line_item.show()
                width = 4 if (getattr(self, '_hovered', False) or self.isSelected()) else 2
                line_item.setPen(QPen(kp1.color, width))
                if kp1.visible_state == 1 or kp2.visible_state == 1:
                    line_item.setOpacity(0.5)
                else:
                    line_item.setOpacity(1.0)
                    
    def update_bounding_box(self):
        if getattr(self, '_is_initializing', False): return
        if not self.kps: return
        
        # Allow keypoints to expand the box if dragged outside
        if not self.kps or getattr(self, '_is_resizing', False): return
        if getattr(self, '_updating_bbox', False): return
        
        min_x = min(kp.pos().x() for kp in self.kps)
        max_x = max(kp.pos().x() for kp in self.kps)
        min_y = min(kp.pos().y() for kp in self.kps)
        max_y = max(kp.pos().y() for kp in self.kps)
        
        kpt_w = max_x - min_x
        kpt_h = max_y - min_y
        
        # YOLO 官方/Ultralytics 标注标准的固定像素边距：
        # 边距在物理上正好是图片高度的 2%！(例如 720p 下为 14.4px)
        img_h = get_current_image_height(self)
        padding = max(img_h * 0.02, 10.0)
        
        new_w = max(kpt_w + padding * 2, 15)
        new_h = max(kpt_h + padding * 2, 15)
        
        target_rect = QRectF(min_x - padding, min_y - padding, new_w, new_h)
        curr_rect = QRectF(-self.box_w/2, -self.box_h/2, self.box_w, self.box_h)
        
        # Check if the bounding box needs to shrink or expand (with a small epsilon to avoid jitter)
        if abs(target_rect.width() - curr_rect.width()) > 0.5 or \
           abs(target_rect.height() - curr_rect.height()) > 0.5 or \
           abs(target_rect.center().x()) > 0.5 or abs(target_rect.center().y()) > 0.5:
            
            self._updating_bbox = True
            
            # The shift in local coordinates to re-center
            shift_x = target_rect.center().x()
            shift_y = target_rect.center().y()
            
            # Move the shape to the new center in scene coordinates
            new_center_scene = self.mapToScene(QPointF(shift_x, shift_y))
            self.setPos(new_center_scene)
            
            # Update box dimensions tightly
            self.box_w = target_rect.width()
            self.box_h = target_rect.height()
            
            # Counter-shift all keypoints so they stay in the same visual place
            for kp in self.kps:
                kp.setPos(kp.pos().x() - shift_x, kp.pos().y() - shift_y)
                
            self.update_geometry()
            self._updating_bbox = False

    def _scale_keypoints(self, old_w, old_h, new_w, new_h):
        if old_w == 0 or old_h == 0: return
        sx = new_w / old_w
        sy = new_h / old_h
        for kp in self.kps:
            kp.setPos(kp.pos().x() * sx, kp.pos().y() * sy)

    def handle_dragged(self, handle_type, scene_pos):
        self._is_resizing = True
        local_pos = self.mapFromScene(scene_pos)
        old_w, old_h = self.box_w, self.box_h

        if handle_type == 'top':
            dy = local_pos.y() - (-self.box_h / 2)
            if self.box_h - dy < 5: dy = self.box_h - 5
            self.box_h -= dy
            scene_offset = self.mapToScene(QPointF(0, dy / 2)) - self.mapToScene(QPointF(0, 0))
            self.setPos(self.pos() + scene_offset)
            self._scale_keypoints(old_w, old_h, self.box_w, self.box_h)

        elif handle_type == 'bottom':
            dy = local_pos.y() - (self.box_h / 2)
            if self.box_h + dy < 5: dy = -(self.box_h - 5)
            self.box_h += dy
            scene_offset = self.mapToScene(QPointF(0, dy / 2)) - self.mapToScene(QPointF(0, 0))
            self.setPos(self.pos() + scene_offset)
            self._scale_keypoints(old_w, old_h, self.box_w, self.box_h)

        elif handle_type == 'left':
            dx = local_pos.x() - (-self.box_w / 2)
            if self.box_w - dx < 5: dx = self.box_w - 5
            self.box_w -= dx
            scene_offset = self.mapToScene(QPointF(dx / 2, 0)) - self.mapToScene(QPointF(0, 0))
            self.setPos(self.pos() + scene_offset)
            self._scale_keypoints(old_w, old_h, self.box_w, self.box_h)

        elif handle_type == 'right':
            dx = local_pos.x() - (self.box_w / 2)
            if self.box_w + dx < 5: dx = -(self.box_w - 5)
            self.box_w += dx
            scene_offset = self.mapToScene(QPointF(dx / 2, 0)) - self.mapToScene(QPointF(0, 0))
            self.setPos(self.pos() + scene_offset)
            self._scale_keypoints(old_w, old_h, self.box_w, self.box_h)

        elif handle_type in ['tl', 'tr', 'bl', 'br']:
            # For corners, we scale proportionally or freeform
            dx = local_pos.x() - (-self.box_w / 2 if 'l' in handle_type else self.box_w / 2)
            dy = local_pos.y() - (-self.box_h / 2 if 't' in handle_type else self.box_h / 2)
            
            if 'l' in handle_type:
                if self.box_w - dx < 5: dx = self.box_w - 5
                self.box_w -= dx
            else:
                if self.box_w + dx < 5: dx = -(self.box_w - 5)
                self.box_w += dx
                
            if 't' in handle_type:
                if self.box_h - dy < 5: dy = self.box_h - 5
                self.box_h -= dy
            else:
                if self.box_h + dy < 5: dy = -(self.box_h - 5)
                self.box_h += dy

            cx_off = dx / 2 if 'l' in handle_type else dx / 2
            cy_off = dy / 2 if 't' in handle_type else dy / 2
            scene_offset = self.mapToScene(QPointF(cx_off, cy_off)) - self.mapToScene(QPointF(0, 0))
            self.setPos(self.pos() + scene_offset)
            self._scale_keypoints(old_w, old_h, self.box_w, self.box_h)

        elif handle_type == 'rotate':
            center_scene = self.mapToScene(QPointF(0, 0))
            dx = scene_pos.x() - center_scene.x()
            dy = scene_pos.y() - center_scene.y()
            angle_deg = math.degrees(math.atan2(dy, dx))
            self.setRotation(angle_deg + 90)

        self.update_geometry()
        self._is_resizing = False

    def set_hover_state(self, state):
        self._hovered = state
        if not self.isSelected():
            if state:
                self.setZValue(50.0)
                # 未选中时，靠近骨架线/关键点显示手指光标，提示可点击进入编辑
                self.setCursor(Qt.PointingHandCursor)
                # 悬停显示浅绿色虚线提示框
                self.rect_item.setPen(QPen(QColor(50, 255, 50, 150), 2, Qt.DashLine))
                self.rect_item.show()
            else:
                self.setZValue(0.0)
                self.unsetCursor()
                self.rect_item.setPen(QPen(Qt.NoPen))
                self.rect_item.hide()
        else:
            if state:
                self.setCursor(Qt.OpenHandCursor)
                self.rect_item.setPen(QPen(QColor(50, 255, 50, 255), 2))
            else:
                self.unsetCursor()
                self.rect_item.setPen(QPen(QColor(50, 255, 50, 200), 2))
                self.rect_item.show()

        # Update visual size of keypoints
        scale_factor = 1.5 if (state or self.isSelected()) else 1.0
        for kp in self.kps:
            kp.setScale(scale_factor)

        self.update_lines()
        self._update_handle_visibility()
        self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=state)

    def hoverEnterEvent(self, event):
        self.set_hover_state(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.set_hover_state(False)
        super().hoverLeaveEvent(event)
    def _update_handle_visibility(self):
        if self.is_temp: return
        # 默认不显示外框和手柄，只有在选中状态且被 hovered 或处于编辑模式时显示
        # 严格控制编辑模式
        visible = self.isSelected()
        
        # 控制矩形边框的显示
        if visible or getattr(self, '_hovered', False):
            self.rect_item.show()
        else:
            self.rect_item.hide()
            
        if visible:
            self.rotate_line.show()
        else:
            self.rotate_line.hide()

        for h in self.handles:
            h.setVisible(visible)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and not getattr(self, 'is_temp', False) and not getattr(self, '_is_resizing', False):
            return super().itemChange(change, clamp_item_position(self, value, overflow_ratio=0.5))
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.setZValue(100.0 if self.isSelected() else 0.0)
            self.set_hover_state(getattr(self, '_hovered', False))
        elif change == QGraphicsItem.ItemPositionHasChanged:
            if not self.is_temp:
                self.update_label_position(self)
        return super().itemChange(change, value)


class PolyShape(QGraphicsPolygonItem, BaseShape):
    def __init__(self, polygon, label="", is_temp=False):
        super().__init__(polygon)
        self.label = label
        self.is_temp = is_temp
        self.handles = []
        self._updating_handles = False
        self._hovered = False
        self._dragging_edge_idx = -1

        self.ghost_idx = -1
        self.ghost_pos = None
        self.ghost_handle = QGraphicsEllipseItem(-3.5, -3.5, 7, 7, self)
        self.ghost_handle.setBrush(QBrush(QColor(255, 255, 255, 180)))
        self.ghost_handle.setPen(QPen(QColor(28, 126, 214, 150), 1.5))
        self.ghost_handle.setZValue(999)
        self.ghost_handle.setAcceptedMouseButtons(Qt.NoButton)
        self.ghost_handle.hide()

        if is_temp:
            self.setPen(QPen(QColor(28, 126, 214), 2, Qt.DashLine))
            self.setBrush(QBrush(QColor(28, 126, 214, 50)))
        else:
            self.setup_style(self)
            self.setup_label(self)
            if label:
                self.update_label_text(label)
                self.update_label_position(self)

        self.update_handles()

    def hoverEnterEvent(self, event):
        self.apply_hover_enter(self)
        self._hovered = True
        self._update_handle_visibility()
        if not self.is_temp:
            self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=True)
        super().hoverEnterEvent(event)

    def hoverMoveEvent(self, event):
        if self.is_temp or self._dragging_edge_idx != -1 or not self.isSelected():
            super().hoverMoveEvent(event)
            return

        pt = event.pos()
        poly = self.polygon()
        min_dist = float('inf')
        insert_idx = -1
        closest_pt = None

        for i in range(poly.count()):
            p1 = poly[i]
            p2 = poly[(i + 1) % poly.count()]
            dist, proj = point_to_segment_dist(pt, p1, p2)
            if dist < min_dist:
                min_dist = dist
                insert_idx = i + 1
                closest_pt = proj

        if min_dist < 8:
            self.ghost_idx = insert_idx
            self.ghost_pos = closest_pt
            self.ghost_handle.setPos(closest_pt)
            self.ghost_handle.show()
            self.setCursor(Qt.CrossCursor)
        else:
            self.ghost_idx = -1
            self.ghost_handle.hide()
            self.setCursor(Qt.PointingHandCursor)

        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.apply_hover_leave(self)
        self._hovered = False
        self.ghost_idx = -1
        self.ghost_handle.hide()

        if not any(h.isUnderMouse() for h in self.handles):
            self._update_handle_visibility()

        if not self.is_temp:
            self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=False)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.ghost_idx != -1 and self.isSelected():
            self._dragging_edge_idx = self.ghost_idx
            poly = self.polygon()
            poly.insert(self.ghost_idx, self.ghost_pos)
            self.setPolygon(poly)
            self.ghost_handle.hide()
            self.ghost_idx = -1

            self.setFlag(QGraphicsItem.ItemIsMovable, False)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging_edge_idx != -1:
            poly = self.polygon()

            scene_pos = self.mapToScene(event.pos())
            scene = self.scene()
            if scene:
                rect = scene.sceneRect()
                clamped_x = max(rect.left(), min(scene_pos.x(), rect.right()))
                clamped_y = max(rect.top(), min(scene_pos.y(), rect.bottom()))
                clamped_local = self.mapFromScene(QPointF(clamped_x, clamped_y))
                poly[self._dragging_edge_idx] = clamped_local
            else:
                poly[self._dragging_edge_idx] = event.pos()

            self.setPolygon(poly)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging_edge_idx != -1:
            self._dragging_edge_idx = -1
            self.setFlag(QGraphicsItem.ItemIsMovable, True)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def remove_handle(self, handle):
        if len(self.handles) <= 3:
            return
        idx = self.handles.index(handle)
        poly = self.polygon()
        poly.remove(idx)
        self.setPolygon(poly)

    def _update_handle_visibility(self):
        if self.is_temp: return
        scene = self.scene()
        is_drawing_poly = False
        if scene and getattr(scene, 'mode', None) == 2:
            if len(getattr(scene, 'poly_pts', [])) > 0:
                is_drawing_poly = True
                
        visible = self.isSelected() or (self._hovered and not is_drawing_poly)
        for h in self.handles:
            h.setVisible(visible)

    def update_handles(self):
        polygon = self.polygon()
        while len(self.handles) < polygon.count():
            handle = HandleItem(self)
            if self.is_temp:
                handle.setFlag(QGraphicsItem.ItemIsMovable, False)
                handle.setFlag(QGraphicsItem.ItemSendsGeometryChanges, False)
                handle.hide()
            self.handles.append(handle)

        while len(self.handles) > polygon.count():
            h = self.handles.pop()
            h.setParentItem(None)
            if self.scene(): self.scene().removeItem(h)

        self._updating_handles = True
        for i, handle in enumerate(self.handles):
            handle.setPos(polygon[i])
        self._update_handle_visibility()
        self._updating_handles = False

    def setPolygon(self, polygon):
        super().setPolygon(polygon)
        self.update_handles()
        if not self.is_temp:
            self.update_label_position(self)

    def update_from_handles(self):
        if self.is_temp or self._updating_handles: return
        polygon = QPolygonF()
        for handle in self.handles:
            polygon.append(handle.pos())
        super().setPolygon(polygon)
        self.update_label_position(self)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and not getattr(self, 'is_temp', False):
            return super().itemChange(change, clamp_item_position(self, value))

        if not self.is_temp:
            if change == QGraphicsItem.ItemSelectedHasChanged:
                self.setZValue(100.0 if self.isSelected() else 0.0)
                self._update_handle_visibility()
                self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=self._hovered)
                if not self.isSelected():
                    self.ghost_idx = -1
                    if hasattr(self, 'ghost_handle'):
                        self.ghost_handle.hide()
            elif change == QGraphicsItem.ItemPositionHasChanged:
                self.update_label_position(self)
        return super().itemChange(change, value)


class PointShape(QGraphicsEllipseItem, BaseShape):
    def __init__(self, point, label=""):
        r = 4
        super().__init__(point.x() - r, point.y() - r, r * 2, r * 2)
        self.normal_pen = QPen(QColor(250, 82, 82), 2)
        self.normal_brush = QBrush(QColor(250, 82, 82, 150))
        self.hover_brush = QBrush(QColor(250, 82, 82, 220))
        self.setPen(self.normal_pen)
        self.setBrush(self.normal_brush)
        self.setFlags(
            QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.label = label

        self.setup_label(self)
        if label:
            self.update_label_text(label)
            self.update_label_position(self)

    def set_color(self, color):
        """dynamically update point color based on class color"""
        self.normal_pen = QPen(color, 2)
        self.normal_brush = QBrush(QColor(color.red(), color.green(), color.blue(), 150))
        self.hover_brush = QBrush(QColor(color.red(), color.green(), color.blue(), 220))
        self.setPen(self.normal_pen)
        self.setBrush(self.normal_brush)
        if hasattr(self, 'label_text') and self.label_text:
            self.label_text.setDefaultTextColor(color)

    def hoverEnterEvent(self, event):
        self.setBrush(self.hover_brush)
        self.setCursor(Qt.PointingHandCursor)
        self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=True)
        super().hoverEnterEvent(event)
        scene = self.scene()
        if scene and hasattr(scene, 'canvas_item_hovered'):
            scene.canvas_item_hovered.emit(self, True)

    def hoverLeaveEvent(self, event):
        self.setBrush(self.normal_brush)
        self.setCursor(Qt.ArrowCursor)
        self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=False)
        super().hoverLeaveEvent(event)
        scene = self.scene()
        if scene and hasattr(scene, 'canvas_item_hovered'):
            scene.canvas_item_hovered.emit(self, False)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            return super().itemChange(change, clamp_item_position(self, value))

        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.update_label_visibility(self, is_selected=self.isSelected(), is_hovered=self.isUnderMouse())
        elif change == QGraphicsItem.ItemPositionHasChanged:
            self.update_label_position(self)
        return super().itemChange(change, value)


class OBBHandle(QGraphicsItem):
    """自定义胶囊形/圆形的 OBB 操作手柄 (接管底层鼠标事件)"""

    def __init__(self, handle_type, parent):
        super().__init__(parent)
        self.handle_type = handle_type  # 'top', 'bottom', 'left', 'right', 'rotate', 'tl', 'tr', 'bl', 'br'
        self.setAcceptHoverEvents(True)
        self.setZValue(100)

        self.w, self.h = 0, 0
        if self.handle_type in ['top', 'bottom']:
            self.w, self.h = 16, 6  # 横向胶囊
        elif self.handle_type in ['left', 'right']:
            self.w, self.h = 6, 16  # 纵向胶囊
        elif self.handle_type in ['rotate', 'tl', 'tr', 'bl', 'br']:
            self.w, self.h = 10, 10  # 旋转圆球和角落圆球

    def boundingRect(self):
        return QRectF(-self.w / 2, -self.h / 2, self.w, self.h)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(28, 126, 214), 2))
        if self.handle_type in ['rotate', 'tl', 'tr', 'bl', 'br']:
            painter.drawEllipse(self.boundingRect())
        else:
            painter.drawRoundedRect(self.boundingRect(), min(self.w, self.h) / 2, min(self.w, self.h) / 2)

    def hoverEnterEvent(self, event):
        if self.handle_type == 'rotate':
            self.setCursor(Qt.OpenHandCursor)
        elif self.handle_type in ['tl', 'br']:
            self.setCursor(Qt.SizeFDiagCursor)
        elif self.handle_type in ['tr', 'bl']:
            self.setCursor(Qt.SizeBDiagCursor)
        else:
            self.setCursor(Qt.CrossCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().hoverLeaveEvent(event)

    # ================= 鼠标事件接管 =================
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.handle_type == 'rotate':
                self.setCursor(Qt.ClosedHandCursor)

            # 记录当前正在拖拽的手柄，并锁定父容器不被整体拖走
            self.parentItem()._dragging_handle = self.handle_type
            self.parentItem().setFlag(QGraphicsItem.ItemIsMovable, False)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.parentItem()._dragging_handle == self.handle_type:
            # 将拖拽产生的全局坐标，实时发送给父容器进行解算
            self.parentItem().handle_dragged(self.handle_type, event.scenePos())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.parentItem()._dragging_handle = None
            # 解除父容器锁定
            self.parentItem().setFlag(QGraphicsItem.ItemIsMovable, True)
            if self.handle_type == 'rotate':
                self.setCursor(Qt.OpenHandCursor)
                
            # 通知主场景更新状态（保存撤销记录）
            if not getattr(self.parentItem(), 'is_temp', False) and self.scene() and hasattr(self.scene(), 'state_changed'):
                self.scene().state_changed.emit()
                
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class RotatedRectShape(QGraphicsObject, BaseShape):
    def __init__(self, cx, cy, w, h, angle, label="", is_temp=False):
        super().__init__()
        self.is_temp = is_temp
        self.label = label

        # 用于区分“整体移动”和“手柄拉伸”
        self._is_resizing = False

        # 依赖宽高
        self.box_w = w
        self.box_h = h
        self.setPos(cx, cy)
        self.setRotation(angle)

        self.setFlags(
            QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)

        self._dragging_handle = None
        self._hovered = False

        self.rect_item = QGraphicsRectItem(self)
        if is_temp:
            self.rect_item.setPen(QPen(QColor(0, 255, 0), 2, Qt.DashLine))
            self.rect_item.setBrush(QBrush(QColor(0, 255, 0, 50)))
        else:
            self.rect_item.setPen(QPen(QColor(28, 126, 214), 2))
            self.rect_item.setBrush(QBrush(QColor(28, 126, 214, 50)))

        self._base_color = QColor(28, 126, 214)

        self.rotate_line = QGraphicsLineItem(self)
        self.rotate_line.setPen(QPen(QColor(28, 126, 214), 1.5, Qt.DashLine))

        self.h_top = OBBHandle('top', self)
        self.h_bottom = OBBHandle('bottom', self)
        self.h_left = OBBHandle('left', self)
        self.h_right = OBBHandle('right', self)
        self.h_rotate = OBBHandle('rotate', self)

        self.handles = [self.h_top, self.h_bottom, self.h_left, self.h_right, self.h_rotate, self.rotate_line]

        self.update_geometry()
        self._update_handle_visibility()

        if not is_temp:
            self.setup_label(self)
            if label: self.update_label_text(label)

        if is_temp:
            for h in self.handles:
                h.hide()

    def set_color(self, color):
        """dynamically update OBB color based on class color"""
        self._base_color = color
        if not self.is_temp:
            self.rect_item.setPen(QPen(color, 2))
            self.rect_item.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
            self.rotate_line.setPen(QPen(color, 1.5, Qt.DashLine))
        if hasattr(self, 'label_text') and self.label_text:
            self.label_text.setDefaultTextColor(color)

    def boundingRect(self):
        r = self.box_h / 2 + 35
        return QRectF(-self.box_w / 2 - 10, -r, self.box_w + 20, r + self.box_h / 2 + 10)

    def paint(self, painter, option, widget=None):
        pass

    def update_geometry(self):
        w, h = self.box_w, self.box_h
        self.rect_item.setRect(-w / 2, -h / 2, w, h)

        self.h_top.setPos(0, -h / 2)
        self.h_bottom.setPos(0, h / 2)
        self.h_left.setPos(-w / 2, 0)
        self.h_right.setPos(w / 2, 0)

        self.h_rotate.setPos(0, -h / 2 - 30)
        self.rotate_line.setLine(0, -h / 2, 0, -h / 2 - 30)

        self.prepareGeometryChange()
        self.update_label_position(self)

    def handle_dragged(self, handle_type, scene_pos):
        self._is_resizing = True

        local_pos = self.mapFromScene(scene_pos)
        if handle_type == 'top':
            dy = local_pos.y() - (-self.box_h / 2)
            if self.box_h - dy < 5: dy = self.box_h - 5
            self.box_h -= dy
            scene_offset = self.mapToScene(QPointF(0, dy / 2)) - self.mapToScene(QPointF(0, 0))
            self.setPos(self.pos() + scene_offset)

        elif handle_type == 'bottom':
            dy = local_pos.y() - (self.box_h / 2)
            if self.box_h + dy < 5: dy = -(self.box_h - 5)
            self.box_h += dy
            scene_offset = self.mapToScene(QPointF(0, dy / 2)) - self.mapToScene(QPointF(0, 0))
            self.setPos(self.pos() + scene_offset)

        elif handle_type == 'left':
            dx = local_pos.x() - (-self.box_w / 2)
            if self.box_w - dx < 5: dx = self.box_w - 5
            self.box_w -= dx
            scene_offset = self.mapToScene(QPointF(dx / 2, 0)) - self.mapToScene(QPointF(0, 0))
            self.setPos(self.pos() + scene_offset)

        elif handle_type == 'right':
            dx = local_pos.x() - (self.box_w / 2)
            if self.box_w + dx < 5: dx = -(self.box_w - 5)
            self.box_w += dx
            scene_offset = self.mapToScene(QPointF(dx / 2, 0)) - self.mapToScene(QPointF(0, 0))
            self.setPos(self.pos() + scene_offset)

        elif handle_type == 'rotate':
            center_scene = self.mapToScene(QPointF(0, 0))
            dx = scene_pos.x() - center_scene.x()
            dy = scene_pos.y() - center_scene.y()
            angle_deg = math.degrees(math.atan2(dy, dx))
            self.setRotation(angle_deg + 90)

        self.update_geometry()
        self._is_resizing = False

    def polygon(self):
        w, h = self.box_w, self.box_h
        pts = [
            QPointF(-w / 2, -h / 2),
            QPointF(w / 2, -h / 2),
            QPointF(w / 2, h / 2),
            QPointF(-w / 2, h / 2)
        ]
        return QPolygonF([self.mapToScene(p) for p in pts])

    def hoverEnterEvent(self, event):
        self._hovered = True
        scene = self.scene()
        is_drawing_poly = False
        if scene and getattr(scene, 'mode', None) == 2:
            if len(getattr(scene, 'poly_pts', [])) > 0:
                is_drawing_poly = True
                
        if not is_drawing_poly:
            if not self.isSelected():
                self.setZValue(50.0)
            c = self._base_color
            self.rect_item.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), 120)))
            if scene and hasattr(scene, 'canvas_item_hovered'):
                scene.canvas_item_hovered.emit(self, True)
        self._update_handle_visibility()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False
        if not self.isSelected():
            self.setZValue(0.0)
        c = self._base_color
        self.rect_item.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), 50)))
        self._update_handle_visibility()
        super().hoverLeaveEvent(event)
        scene = self.scene()
        if scene and hasattr(scene, 'canvas_item_hovered'):
            scene.canvas_item_hovered.emit(self, False)

    def _update_handle_visibility(self):
        if self.is_temp: return
        scene = self.scene()
        is_drawing_poly = False
        if scene and getattr(scene, 'mode', None) == 2:
            if len(getattr(scene, 'poly_pts', [])) > 0:
                is_drawing_poly = True
                
        visible = self.isSelected() or (self._hovered and not is_drawing_poly)
        for h in self.handles:
            h.setVisible(visible)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and not getattr(self, 'is_temp', False) and not getattr(self, '_is_resizing', False):
            scene = self.scene()
            if scene:
                rect = scene.sceneRect()
                new_pos = value
                valid_x = max(rect.left(), min(new_pos.x(), rect.right()))
                valid_y = max(rect.top(), min(new_pos.y(), rect.bottom()))
                return QPointF(valid_x, valid_y)

        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.setZValue(100.0 if self.isSelected() else 0.0)
            self._update_handle_visibility()
        elif change == QGraphicsItem.ItemPositionHasChanged:
            self.update_label_position(self)
        return super().itemChange(change, value)