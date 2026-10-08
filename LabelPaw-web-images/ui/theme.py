# ui/theme.py

DARK_THEME = """
/* LabelPaw Dark Mode Theme */
QMainWindow, QWidget {
    background-color: #020617;
    color: #F8FAFC;
    font-family: "Fira Code", "Fira Sans", "Microsoft YaHei", sans-serif;
}

QToolBar {
    background-color: #0F172A;
    border: none;
    padding: 8px;
    spacing: 12px;
}

QToolButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 8px 8px 8px 12px;
    color: #94A3B8;
    font-size: 13px;
    font-weight: 600;
    text-align: left;
}
QToolButton::icon {
    padding-right: 8px;
}
QToolButton:hover {
    background-color: #1E293B;
    color: #F8FAFC;
}
QToolButton:checked {
    background-color: #1E293B;
    color: #22C55E;
}

QDockWidget {
    font-weight: bold;
    color: #F8FAFC;
}
QDockWidget::title {
    background: #0F172A;
    padding: 12px;
    border-bottom: 1px solid #334155;
}

QListWidget {
    background-color: #0F172A;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 4px;
    outline: 0;
}
QListWidget::item {
    padding: 4px 8px;
    border-radius: 4px;
    color: #cbd5e1;
    font-size: 13px;
    min-height: 24px;
}
QListWidget::item:hover { background-color: #1E293B; }
QListWidget::item:selected {
    background-color: #1E293B;
    color: #22C55E;
}
QListWidget QLineEdit {
    background-color: #1E293B;
    color: #F8FAFC;
    border: 1px solid #22C55E;
    border-radius: 4px;
    padding: 1px 4px;
    margin: 0px;
    font-size: 13px;
    min-height: 20px;
}

QGraphicsView {
    background-color: #000000;
    border: 1px solid #334155;
    border-radius: 8px;
    margin: 8px;
}

QLineEdit, QPushButton {
    padding: 8px 12px;
    border: 1px solid #334155;
    border-radius: 8px;
    background: #0F172A;
    color: #F8FAFC;
    font-weight: 500;
}
QLineEdit:focus { border: 1px solid #22C55E; background: #1E293B; }
QPushButton:hover { background: #1E293B; border: 1px solid #475569; }
QPushButton:pressed { background: #334155; }

QStatusBar {
    background-color: #0F172A;
    border-top: 1px solid #334155;
}
QStatusBar QLabel { color: #94A3B8; }
QLabel { color: #cbd5e1; background: transparent; }

#topBar {
    background-color: #0F172A;
    border-bottom: 1px solid #334155;
}

#btnTopBar {
    background-color: transparent;
    border: none;
    color: #94A3B8;
    padding: 0px;
    border-radius: 18px;
}
#btnTopBar:hover {
    background-color: #1E293B;
    color: #F8FAFC;
}

#logoWidget {
    background-color: transparent;
    border: none;
}
#logoLabel {
    background-color: transparent;
}

#canvasArea {
    background-color: #020617;
}
#rightPanel {
    background-color: #0F172A;
    border-left: 1px solid #334155;
}
#classesContainer, #filesContainer, #rightSplitter, #filesHeader {
    background-color: transparent;
}
#rightPanelTitle {
    color: #F8FAFC;
    padding: 4px 0px;
}
#contentSplitter {
    background-color: #020617;
}

/* Specific styles for Top Toolbar floating effect */
#annotationToolbar {
    background-color: #0F172A;
    border: 1px solid #334155;
    border-radius: 12px;
}
#annotationToolbar QPushButton {
    background-color: transparent;
    border: none;
    color: #94A3B8;
    font-weight: bold;
    padding: 6px 12px;
}
#annotationToolbar QPushButton:hover {
    background-color: #1E293B;
    color: #F8FAFC;
    border-radius: 6px;
}
#annotationToolbar QPushButton:checked {
    background-color: #1E293B;
    color: #22C55E;
    border-radius: 6px;
}

/* Format Selector */
#formatBtn {
    background-color: transparent;
    border: 1px solid transparent;
    color: #F8FAFC;
    font-size: 13px;
    font-weight: bold;
    padding: 6px;
    border-radius: 8px;
    text-align: left;
}
#formatBtn:hover {
    background-color: #1E293B;
    color: #22C55E;
}
#formatBtn::menu-indicator {
    image: none;
}
#formatMenu {
    background-color: #0F172A;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 6px 0px;
}
#formatMenu::item {
    padding: 8px 36px 8px 32px;
    margin: 2px 6px;
    border-radius: 4px;
    color: #cbd5e1;
    font-size: 13px;
    font-family: "Microsoft YaHei", sans-serif;
}
#formatMenu::item:selected {
    background-color: #1E293B;
    color: #22C55E;
    font-weight: bold;
}

/* Template Selector */
#templateBtn {
    background-color: transparent;
    border: 1px solid transparent;
    color: #F8FAFC;
    font-size: 13px;
    font-weight: bold;
    padding: 6px 12px;
    border-radius: 8px;
}
#templateBtn:hover {
    background-color: #1E293B;
    color: #22C55E;
}
#templateBtn::menu-indicator {
    image: none;
}
#templateMenu {
    background-color: #0F172A;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 6px 0px;
}
#templateMenu::item {
    padding: 8px 36px 8px 32px;
    margin: 2px 6px;
    border-radius: 4px;
    color: #cbd5e1;
    font-size: 13px;
    font-family: "Microsoft YaHei", sans-serif;
}
#templateMenu::item:selected {
    background-color: #1E293B;
    color: #22C55E;
    font-weight: bold;
}
#templateMenu::separator {
    height: 1px;
    background-color: #334155;
    margin: 4px 0px;
}

/* Bottom Right Widgets */

#samTextGroup {
    background-color: #0F172A;
    border: 1px solid #334155;
    border-radius: 8px;
    margin: 0px;
}
#samTextGroup[focused="true"] {
    border: 1px solid #22C55E;
}

#samPromptInput {
    border: none;
    background-color: transparent;
    color: #F8FAFC;
    font-size: 13px;
    padding: 2px 4px;
}

#samPromptBtn {
    background-color: #22C55E;
    border: none;
    border-radius: 14px;
    padding: 0px;
}
#samPromptBtn:hover {
    background-color: #4ade80;
}
#samPromptBtn:pressed {
    background-color: #16a34a;
}
#samPromptBtn:disabled {
    background-color: #1E293B;
}

#btnOverwrite {
    background-color: transparent;
    border: 1px solid #334155;
    color: #94A3B8;
    border-radius: 11px;
    padding: 2px 8px;
    font-weight: 600;
    font-size: 11px;
}
#btnOverwrite:hover {
    background-color: #1E293B;
    color: #F8FAFC;
    border-color: #475569;
}
#btnOverwrite:checked {
    background-color: rgba(34, 197, 94, 0.15);
    border: 1px solid #22C55E;
    color: #22C55E;
}

#btnDatasetTool {
    background-color: transparent;
    color: #94A3B8;
    border: none;
    border-radius: 6px;
    padding: 8px;
    margin: 4px;
    font-weight: bold;
    font-size: 13px;
    font-family: "Microsoft YaHei";
    text-align: left;
}
#btnDatasetTool:hover {
    background-color: #1E293B;
    color: #22C55E;
}
#btnDatasetTool:pressed {
    background-color: #16a34a;
    color: #020617;
}

#btnModelSelector, #btnPredict, #btnClassFilter {
    border: none;
    background-color: transparent;
    color: #F8FAFC;
    border-radius: 8px;
    padding: 6px 12px;
    font-weight: bold;
    font-size: 14px;
}
#btnModelSelector:hover, #btnPredict:hover, #btnClassFilter:hover { background-color: #334155; }
#btnModelSelector::menu-indicator { image: none; }

#modelMenu {
    background-color: #0F172A;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 6px 0px;
}
#modelMenu::item {
    padding: 8px 36px 8px 32px;
    margin: 2px 6px;
    border-radius: 4px;
    color: #cbd5e1;
    font-size: 13px;
    font-family: "Microsoft YaHei", sans-serif;
}
#modelMenu::item:selected {
    background-color: #1E293B;
    color: #22C55E;
    font-weight: bold;
}

/* Class List Widget */
#classAddBtn {
    background-color: #1E293B;
    color: #22C55E;
    border: 1px solid #334155;
    border-radius: 8px;
    font-size: 16px;
    font-weight: bold;
    padding: 0px;
}
#classAddBtn:hover {
    background-color: #334155;
    border-color: #22C55E;
}
#classSearchInput, #classAddInput {
    background-color: #0F172A;
    color: #F8FAFC;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 13px;
}
#classSearchInput:focus, #classAddInput:focus {
    border: 1px solid #22C55E;
}
#classListWidget {
    background-color: transparent;
    border: none;
    outline: 0;
}
#annotationTreeCard {
    background-color: #0F172A;
    border: 1px solid #334155;
    border-radius: 8px;
}
#classListWidget::item {
    padding: 4px 8px;
    border-radius: 4px;
    color: #cbd5e1;
    font-size: 13px;
    min-height: 26px;
}
#classListWidget::item:hover { background-color: #1E293B; }
#classListWidget::item:selected {
    background-color: #1E293B;
    color: #22C55E;
}
QCheckBox {
    spacing: 5px;
    font-size: 13px;
    font-weight: 500;
}
QCheckBox::indicator, QListWidget::indicator {
    width: 14px;
    height: 14px;
    border: 1.5px solid #475569;
    border-radius: 3px;
    background-color: #0F172A;
}
QCheckBox::indicator:hover, QListWidget::indicator:hover {
    border-color: #22C55E;
}
QCheckBox::indicator:checked, QListWidget::indicator:checked {
    border-color: #22C55E;
    background-color: #22C55E;
    image: url(ui/icon/check.svg);
}
QCheckBox::indicator:indeterminate, QListWidget::indicator:indeterminate {
    border-color: #22C55E;
    background-color: #22C55E;
    image: url(ui/icon/minus.svg);
}
#chkSelectAll {
    color: #64748B;
    font-weight: bold;
    font-size: 12px;
    background-color: transparent;
}
"""

LIGHT_THEME = """
/* LabelPaw Light Mode Theme - Ultralytics Style */
QMainWindow, QWidget {
    background-color: #FFFFFF;
    color: #0F172A;
    font-family: "Fira Code", "Fira Sans", "Microsoft YaHei", sans-serif;
}

QToolBar {
    background-color: #FAFAFA;
    border: none;
    padding: 8px;
    spacing: 12px;
}

QToolButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 8px 8px 8px 12px;
    color: #64748B;
    font-size: 13px;
    font-weight: 600;
    text-align: left;
}
QToolButton::icon {
    padding-right: 8px;
}
QToolButton:hover {
    background-color: #F0F0F0;
    color: #0F172A;
}
QToolButton:checked {
    background-color: #F0F0F0;
    color: #22C55E;
}

QDockWidget {
    font-weight: bold;
    color: #0F172A;
}
QDockWidget::title {
    background: #FFFFFF;
    padding: 12px;
    border-bottom: 1px solid #E2E8F0;
}

QListWidget {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 4px;
    outline: 0;
}
QListWidget::item {
    padding: 4px 8px;
    border-radius: 4px;
    color: #334155;
    font-size: 13px;
    min-height: 24px;
}
QListWidget::item:hover { background-color: #F1F5F9; }
QListWidget::item:selected {
    background-color: #F1F5F9;
    color: #22C55E;
}
QListWidget QLineEdit {
    background-color: #FFFFFF;
    color: #0F172A;
    border: 1px solid #22C55E;
    border-radius: 4px;
    padding: 1px 4px;
    margin: 0px;
    font-size: 13px;
    min-height: 20px;
}

QGraphicsView {
    background-color: #F5F5F5;
    border: 1px solid #E5E5E5;
    border-radius: 8px;
    margin: 8px;
}

QLineEdit, QPushButton {
    padding: 8px 12px;
    border: 1px solid #E5E5E5;
    border-radius: 8px;
    background: #FFFFFF;
    color: #0F172A;
    font-weight: 500;
}
QLineEdit:focus { border: 1px solid #22C55E; background: #FFFFFF; }
QPushButton:hover { background: #F5F5F5; border: 1px solid #D4D4D4; }
QPushButton:pressed { background: #E5E5E5; }

QStatusBar {
    background-color: #FFFFFF;
    border-top: 1px solid #E5E5E5;
}
QStatusBar QLabel { color: #64748B; }
QLabel { color: #334155; background: transparent; }

#topBar {
    background-color: #FFFFFF;
    border-bottom: 1px solid #E5E5E5;
}

#btnTopBar {
    background-color: transparent;
    border: none;
    color: #64748B;
    padding: 0px;
    border-radius: 18px;
}
#btnTopBar:hover {
    background-color: #F0F0F0;
    color: #0F172A;
}

#logoWidget {
    background-color: transparent;
    border: none;
}
#logoLabel {
    background-color: transparent;
}

#canvasArea {
    background-color: #FFFFFF;
}
#rightPanel {
    background-color: #FFFFFF;
    border-left: 1px solid #E5E5E5;
}
#classesContainer, #filesContainer, #rightSplitter, #filesHeader {
    background-color: transparent;
}
#rightPanelTitle {
    color: #0F172A;
    padding: 4px 0px;
}
#contentSplitter {
    background-color: #FFFFFF;
}

/* Specific styles for Top Toolbar floating effect */
#annotationToolbar {
    background-color: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 12px;
}
#annotationToolbar QPushButton {
    background-color: transparent;
    border: none;
    color: #64748B;
    font-weight: bold;
    padding: 6px 12px;
}
#annotationToolbar QPushButton:hover {
    background-color: #F5F5F5;
    color: #0F172A;
    border-radius: 6px;
}
#annotationToolbar QPushButton:checked {
    background-color: #F5F5F5;
    color: #22C55E;
    border-radius: 6px;
}

/* Format Selector */
#formatBtn {
    background-color: transparent;
    border: 1px solid transparent;
    color: #0F172A;
    font-size: 13px;
    font-weight: bold;
    padding: 6px;
    border-radius: 8px;
    text-align: left;
}
#formatBtn:hover {
    background-color: #F1F5F9;
    color: #22C55E;
}
#formatBtn::menu-indicator {
    image: none;
}
#formatMenu {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 6px 0px;
}
#formatMenu::item {
    padding: 8px 36px 8px 32px;
    margin: 2px 6px;
    border-radius: 4px;
    color: #334155;
    font-size: 13px;
    font-family: "Microsoft YaHei", sans-serif;
}
#formatMenu::item:selected {
    background-color: #F1F5F9;
    color: #22C55E;
    font-weight: bold;
}

/* Template Selector */
#templateBtn {
    background-color: transparent;
    border: 1px solid transparent;
    color: #0F172A;
    font-size: 13px;
    font-weight: bold;
    padding: 6px 12px;
    border-radius: 8px;
}
#templateBtn:hover {
    background-color: #F1F5F9;
    color: #22C55E;
}
#templateBtn::menu-indicator {
    image: none;
}
#templateMenu {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 6px 0px;
}
#templateMenu::item {
    padding: 8px 36px 8px 32px;
    margin: 2px 6px;
    border-radius: 4px;
    color: #334155;
    font-size: 13px;
    font-family: "Microsoft YaHei", sans-serif;
}
#templateMenu::item:selected {
    background-color: #F1F5F9;
    color: #22C55E;
    font-weight: bold;
}
#templateMenu::separator {
    height: 1px;
    background-color: #E2E8F0;
    margin: 4px 0px;
}

/* Bottom Right Widgets */

#samTextGroup {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    margin: 0px;
}
#samTextGroup[focused="true"] {
    border: 1px solid #22C55E;
}

#samPromptInput {
    border: none;
    background-color: transparent;
    color: #0F172A;
    font-size: 13px;
    padding: 2px 4px;
}

#samPromptBtn {
    background-color: #22C55E;
    border: none;
    border-radius: 14px;
    padding: 0px;
}
#samPromptBtn:hover {
    background-color: #4ade80;
}
#samPromptBtn:pressed {
    background-color: #16a34a;
}
#samPromptBtn:disabled {
    background-color: #E2E8F0;
}

#btnOverwrite {
    background-color: transparent;
    border: 1px solid #E2E8F0;
    color: #64748B;
    border-radius: 11px;
    padding: 2px 8px;
    font-weight: 600;
    font-size: 11px;
}
#btnOverwrite:hover {
    background-color: #F1F5F9;
    color: #0F172A;
    border-color: #CBD5E1;
}
#btnOverwrite:checked {
    background-color: rgba(34, 197, 94, 0.08);
    border: 1px solid #22C55E;
    color: #16a34a;
}

#btnDatasetTool {
    background-color: transparent;
    color: #64748B;
    border: none;
    border-radius: 6px;
    padding: 8px;
    margin: 4px;
    font-weight: bold;
    font-size: 13px;
    font-family: "Microsoft YaHei";
    text-align: left;
}
#btnDatasetTool:hover {
    background-color: #F1F5F9;
    color: #22C55E;
}
#btnDatasetTool:pressed {
    background-color: #E2E8F0;
    color: #0F172A;
}

#btnModelSelector, #btnPredict, #btnClassFilter {
    border: none;
    background-color: transparent;
    color: #0F172A;
    border-radius: 8px;
    padding: 6px 12px;
    font-weight: bold;
    font-size: 14px;
}
#btnModelSelector:hover, #btnPredict:hover, #btnClassFilter:hover { background-color: #E2E8F0; }
#btnModelSelector::menu-indicator { image: none; }

#modelMenu {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 6px 0px;
}
#modelMenu::item {
    padding: 8px 36px 8px 32px;
    margin: 2px 6px;
    border-radius: 4px;
    color: #334155;
    font-size: 13px;
    font-family: "Microsoft YaHei", sans-serif;
}
#modelMenu::item:selected {
    background-color: #F1F5F9;
    color: #22C55E;
    font-weight: bold;
}

/* Class List Widget */
#classAddBtn {
    background-color: #F1F5F9;
    color: #22C55E;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    font-size: 16px;
    font-weight: bold;
    padding: 0px;
}
#classAddBtn:hover {
    background-color: #E2E8F0;
    border-color: #22C55E;
}
#classSearchInput, #classAddInput {
    background-color: #FFFFFF;
    color: #0F172A;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 13px;
}
#classSearchInput:focus, #classAddInput:focus {
    border: 1px solid #22C55E;
}
#classListWidget {
    background-color: transparent;
    border: none;
    outline: 0;
}
#annotationTreeCard {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
}
#classListWidget::item {
    padding: 4px 8px;
    border-radius: 4px;
    color: #334155;
    font-size: 13px;
    min-height: 26px;
}
#classListWidget::item:hover { background-color: #F1F5F9; }
#classListWidget::item:selected {
    background-color: #F1F5F9;
    color: #22C55E;
}
QCheckBox {
    spacing: 5px;
    font-size: 13px;
    font-weight: 500;
}
QCheckBox::indicator, QListWidget::indicator {
    width: 14px;
    height: 14px;
    border: 1.5px solid #CBD5E1;
    border-radius: 3px;
    background-color: #FFFFFF;
}
QCheckBox::indicator:hover, QListWidget::indicator:hover {
    border-color: #22C55E;
}
QCheckBox::indicator:checked, QListWidget::indicator:checked {
    border-color: #22C55E;
    background-color: #22C55E;
    image: url(ui/icon/check.svg);
}
QCheckBox::indicator:indeterminate, QListWidget::indicator:indeterminate {
    border-color: #22C55E;
    background-color: #22C55E;
    image: url(ui/icon/minus.svg);
}
#chkSelectAll {
    color: #64748B;
    font-weight: bold;
    font-size: 12px;
    background-color: transparent;
}
"""
