"""
类别管理组件 — ClassListWidget
封装搜索、添加、颜色管理的类别列表，供主界面右侧面板使用。
"""

import json
import os
import random

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QListWidget, QListWidgetItem, QColorDialog, QStyledItemDelegate,
    QStyle, QApplication
)
from PySide6.QtCore import Qt, Signal, QRect, QSize, QPoint
from PySide6.QtGui import QColor, QPainter, QBrush, QPen, QFont, QIcon, QPixmap


# ============================================================
# 预定义的高辨识度调色盘（20 种）
# ============================================================
DEFAULT_PALETTE = [
    "#3B82F6",  # Blue
    "#EF4444",  # Red
    "#22C55E",  # Green
    "#F59E0B",  # Amber
    "#8B5CF6",  # Violet
    "#EC4899",  # Pink
    "#06B6D4",  # Cyan
    "#F97316",  # Orange
    "#14B8A6",  # Teal
    "#A855F7",  # Purple
    "#6366F1",  # Indigo
    "#84CC16",  # Lime
    "#E11D48",  # Rose
    "#0EA5E9",  # Sky
    "#D946EF",  # Fuchsia
    "#10B981",  # Emerald
    "#FBBF24",  # Yellow
    "#F43F5E",  # Red-rose
    "#2DD4BF",  # Teal-light
    "#818CF8",  # Indigo-light
]


def get_palette_color(index):
    """从调色盘获取颜色，超出后随机生成高饱和度颜色"""
    if index < len(DEFAULT_PALETTE):
        return QColor(DEFAULT_PALETTE[index])
    # 随机生成高饱和度、中等亮度的颜色
    h = random.randint(0, 359)
    s = random.randint(180, 255)
    v = random.randint(160, 230)
    return QColor.fromHsv(h, s, v)


# ============================================================
# 自定义委托 — 在每行前绘制彩色圆点
# ============================================================
class ColorDotDelegate(QStyledItemDelegate):
    """在 QListWidget 列表项前绘制彩色圆点"""

    DOT_RADIUS = 6
    DOT_LEFT_MARGIN = 8
    TEXT_LEFT_MARGIN = 28  # 圆点 + 间距后的文字起始位置

    def __init__(self, class_widget, parent=None):
        super().__init__(parent)
        self.class_widget = class_widget
        # 追踪鼠标位置以判断是否悬停在圆点上
        if parent:
            parent.setMouseTracking(True)
            parent.viewport().setMouseTracking(True)

    def _is_hovering_dot(self, option):
        """判断鼠标是否在圆点区域"""
        widget = option.widget
        if not widget:
            return False
        cursor_pos = widget.viewport().mapFromGlobal(widget.cursor().pos())
        dot_area_right = option.rect.left() + self.TEXT_LEFT_MARGIN
        return (option.rect.top() <= cursor_pos.y() <= option.rect.bottom()
                and cursor_pos.x() <= dot_area_right)

    def paint(self, painter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制背景（处理 hover / selected 状态）
        self.initStyleOption(option, index)
        # 清空文字，避免默认绘制导致文字重复
        option.text = ""
        style = option.widget.style() if option.widget else QApplication.style()
        # 只绘制背景与选中高亮（文字已清空，不会重复）
        style.drawControl(QStyle.CE_ItemViewItem, option, painter, option.widget)

        # 获取类别名和颜色
        cls_name = index.data(Qt.UserRole) or index.data(Qt.DisplayRole) or ""
        color = self.class_widget.get_class_color(cls_name)

        # 绘制彩色圆点 — 悬停在圆点上时显示空心环
        dot_x = option.rect.left() + self.DOT_LEFT_MARGIN + self.DOT_RADIUS
        dot_y = option.rect.center().y()

        is_hovered = (option.state & QStyle.State_MouseOver) and self._is_hovering_dot(option)

        if is_hovered:
            # 悬停样式：亮色实心 + 略大 + 柔光外圈
            lighter = color.lighter(140)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 40)))
            painter.setPen(QPen(lighter, 1.0))
            painter.drawEllipse(QPoint(dot_x, dot_y), self.DOT_RADIUS + 3, self.DOT_RADIUS + 3)
            painter.setBrush(QBrush(lighter))
            painter.setPen(QPen(Qt.NoPen))
            painter.drawEllipse(QPoint(dot_x, dot_y), self.DOT_RADIUS, self.DOT_RADIUS)
        else:
            # 默认样式：实心圆点
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(130), 1.5))
            painter.drawEllipse(QPoint(dot_x, dot_y), self.DOT_RADIUS, self.DOT_RADIUS)

        # 绘制文字
        text_rect = QRect(
            option.rect.left() + self.TEXT_LEFT_MARGIN,
            option.rect.top(),
            option.rect.width() - self.TEXT_LEFT_MARGIN - 4,
            option.rect.height()
        )

        # 文字颜色：选中时用类别色，未选中跟随主题
        if option.state & QStyle.State_Selected:
            painter.setPen(color)
        else:
            painter.setPen(option.palette.text().color())

        font = option.font
        painter.setFont(font)
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, cls_name)

        painter.restore()

    def createEditor(self, parent, option, index):
        """双击编辑时创建内嵌输入框，右移避开圆点区域"""
        editor = QLineEdit(parent)
        editor.setFrame(False)
        return editor

    def setEditorData(self, editor, index):
        """将当前类别名填入编辑框"""
        text = index.data(Qt.UserRole) or index.data(Qt.DisplayRole) or ""
        editor.setText(text)

    def setModelData(self, editor, model, index):
        """编辑完成后更新数据"""
        model.setData(index, editor.text(), Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        """编辑框位置右移，避开圆点"""
        r = option.rect
        editor.setGeometry(QRect(
            r.left() + self.TEXT_LEFT_MARGIN,
            r.top(),
            r.width() - self.TEXT_LEFT_MARGIN,
            r.height()
        ))

    def sizeHint(self, option, index):
        s = super().sizeHint(option, index)
        return QSize(s.width(), max(s.height(), 30))


# ============================================================
# 主组件 — ClassListWidget
# ============================================================
class ClassListWidget(QWidget):
    """带搜索、添加、颜色管理的类别列表组件"""

    # 对外信号
    class_added = Signal(str)                  # 新类别名
    class_renamed = Signal(str, str)           # (旧名, 新名)
    color_changed = Signal(str, object)        # (类别名, QColor) — 用 object 避免 QColor 注册问题
    item_changed = Signal(QListWidgetItem)     # 兼容旧信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self._class_list = []            # 有序类别名列表
        self._class_colors = {}          # {类别名: QColor}
        self._color_index = 0            # 调色盘分配索引
        self._current_dir = None         # 当前工作目录（用于持久化）

        self._build_ui()
        self._connect_signals()

    # ======================== UI 构建 ========================

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # --- 第一行：搜索输入框 ---
        self.searchInput = QLineEdit()
        self.searchInput.setObjectName("classSearchInput")
        self.searchInput.setPlaceholderText("搜索类别...")
        self.searchInput.setClearButtonEnabled(True)
        layout.addWidget(self.searchInput)

        # --- 第二行：添加按钮 + 添加输入框 ---
        add_bar = QHBoxLayout()
        add_bar.setContentsMargins(0, 0, 0, 0)
        add_bar.setSpacing(4)

        self.btnAdd = QPushButton("+")
        self.btnAdd.setObjectName("classAddBtn")
        self.btnAdd.setFixedSize(30, 30)
        self.btnAdd.setCursor(Qt.PointingHandCursor)
        self.btnAdd.setToolTip("添加新类别")

        self.addInput = QLineEdit()
        self.addInput.setObjectName("classAddInput")
        self.addInput.setPlaceholderText("输入新类别名...")

        add_bar.addWidget(self.btnAdd)
        add_bar.addWidget(self.addInput, 1)
        layout.addLayout(add_bar)

        # --- 类别列表 ---
        self.listWidget = QListWidget()
        self.listWidget.setObjectName("classListWidget")
        self.listWidget.setItemDelegate(ColorDotDelegate(self, self.listWidget))
        layout.addWidget(self.listWidget, 1)

    def _connect_signals(self):
        self.searchInput.textChanged.connect(self._on_search_text_changed)
        self.addInput.returnPressed.connect(self._on_add_class)
        self.btnAdd.clicked.connect(self._on_add_class)
        self.listWidget.clicked.connect(self._on_list_clicked)
        self.listWidget.itemChanged.connect(self._on_item_changed)

    # ======================== 搜索过滤 ========================

    def _on_search_text_changed(self, text):
        """实时过滤列表"""
        keyword = text.strip().lower()
        for i in range(self.listWidget.count()):
            item = self.listWidget.item(i)
            cls_name = item.data(Qt.UserRole) or item.text()
            item.setHidden(keyword != "" and keyword not in cls_name.lower())

    # ======================== 添加类别 ========================

    def _on_add_class(self):
        """通过添加输入框添加新类别"""
        text = self.addInput.text().strip()
        if not text:
            return
        if text in self._class_list:
            # 已存在则高亮选中
            for i in range(self.listWidget.count()):
                item = self.listWidget.item(i)
                if (item.data(Qt.UserRole) or item.text()) == text:
                    self.listWidget.setCurrentItem(item)
                    break
            self.addInput.clear()
            return

        self.add_class(text)
        self.addInput.clear()
        self.class_added.emit(text)

    # ======================== 颜色点击 ========================

    def _on_list_clicked(self, index):
        """检测是否点击了颜色圆点区域"""
        item = self.listWidget.itemFromIndex(index)
        if not item:
            return

        # 获取点击的实际位置
        cursor_pos = self.listWidget.viewport().mapFromGlobal(self.listWidget.cursor().pos())
        item_rect = self.listWidget.visualItemRect(item)

        # 圆点区域：左侧 0 ~ TEXT_LEFT_MARGIN 的范围
        dot_area_right = item_rect.left() + ColorDotDelegate.TEXT_LEFT_MARGIN
        if cursor_pos.x() <= dot_area_right:
            cls_name = item.data(Qt.UserRole) or item.text()
            current_color = self.get_class_color(cls_name)
            new_color = QColorDialog.getColor(current_color, self, f"选择 '{cls_name}' 的颜色")
            if new_color.isValid():
                self._class_colors[cls_name] = new_color
                self.listWidget.viewport().update()
                self.color_changed.emit(cls_name, new_color)
                self._save_colors()

    # ======================== 重命名 ========================

    def _on_item_changed(self, item):
        """处理列表项文字编辑（重命名）"""
        new_name = item.text().strip()
        old_name = item.data(Qt.UserRole)

        if not old_name or new_name == old_name:
            return

        if not new_name:
            item.setText(old_name)
            return

        if new_name in self._class_list:
            item.setText(old_name)
            return

        # 更新内部数据
        idx = self._class_list.index(old_name)
        self._class_list[idx] = new_name
        item.setData(Qt.UserRole, new_name)

        # 同步颜色映射
        if old_name in self._class_colors:
            self._class_colors[new_name] = self._class_colors.pop(old_name)
            self._save_colors()

        self.class_renamed.emit(old_name, new_name)
        self.item_changed.emit(item)

    # ======================== 公共 API ========================

    def add_class(self, cls_name, color=None):
        """添加一个类别（带颜色），如果已存在则跳过"""
        if cls_name in self._class_list:
            return
        self._class_list.append(cls_name)

        # 分配颜色
        if color and isinstance(color, QColor):
            self._class_colors[cls_name] = color
        elif cls_name not in self._class_colors:
            self._class_colors[cls_name] = get_palette_color(self._color_index)
            self._color_index += 1

        # 添加列表项
        item = QListWidgetItem(cls_name)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        item.setData(Qt.UserRole, cls_name)
        self.listWidget.addItem(item)

    def remove_class(self, cls_name):
        """移除一个类别"""
        if cls_name not in self._class_list:
            return
        self._class_list.remove(cls_name)
        self._class_colors.pop(cls_name, None)
        for i in range(self.listWidget.count()):
            item = self.listWidget.item(i)
            if (item.data(Qt.UserRole) or item.text()) == cls_name:
                self.listWidget.takeItem(i)
                break

    def clear_classes(self):
        """清空所有类别"""
        self._class_list.clear()
        self._class_colors.clear()
        self._color_index = 0
        self.listWidget.clear()

    def get_class_list(self):
        """返回有序的类别名列表"""
        return list(self._class_list)

    def get_class_color(self, cls_name):
        """获取类别颜色，不存在则返回默认蓝色"""
        return self._class_colors.get(cls_name, QColor("#3B82F6"))

    def get_all_colors(self):
        """返回所有类别颜色的字典副本"""
        return dict(self._class_colors)

    def set_class_color(self, cls_name, color):
        """外部设置类别颜色"""
        self._class_colors[cls_name] = color
        self.listWidget.viewport().update()

    # ======================== 持久化 ========================

    def set_working_dir(self, dir_path):
        """设置工作目录（用于 class_colors.json 的读写）"""
        self._current_dir = dir_path

    def load_classes(self, dir_path):
        """从目录中加载 classes.txt 和 class_colors.json"""
        self.clear_classes()
        self._current_dir = dir_path
        self._color_index = 0

        # 先加载颜色映射
        color_file = os.path.join(dir_path, "class_colors.json")
        saved_colors = {}
        if os.path.exists(color_file):
            try:
                with open(color_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    saved_colors = {k: QColor(v) for k, v in raw.items()}
            except Exception:
                pass

        # 加载类别列表
        class_file = os.path.join(dir_path, "classes.txt")
        if os.path.exists(class_file):
            with open(class_file, "r", encoding="utf-8") as f:
                for line in f:
                    cls_name = line.strip()
                    if cls_name:
                        color = saved_colors.get(cls_name, None)
                        self.add_class(cls_name, color)

    def save_classes(self):
        """保存 classes.txt 和 class_colors.json"""
        if not self._current_dir:
            return
        # 保存 classes.txt
        class_file = os.path.join(self._current_dir, "classes.txt")
        with open(class_file, "w", encoding="utf-8") as f:
            f.write("\n".join(self._class_list))
        self._save_colors()

    def _save_colors(self):
        """仅保存颜色映射"""
        if not self._current_dir:
            return
        color_file = os.path.join(self._current_dir, "class_colors.json")
        try:
            data = {k: v.name() for k, v in self._class_colors.items()}
            with open(color_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
