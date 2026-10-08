from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt

class AuthorInfoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("作者信息")
        self.resize(400, 280)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint | Qt.WindowCloseButtonHint)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Author Name
        lbl_author = QLabel("<b>作者：</b>落花不写码")
        lbl_author.setTextFormat(Qt.RichText)
        lbl_author.setOpenExternalLinks(True)
        
        # CSDN
        lbl_csdn = QLabel('<b>CSDN主页：</b><a href="https://blog.csdn.net/weixin_44779079?spm=1010.2135.3001.10640">点击访问</a>')
        lbl_csdn.setTextFormat(Qt.RichText)
        lbl_csdn.setOpenExternalLinks(True)
        
        # Bilibili
        lbl_bilibili = QLabel('<b>B站主页：</b><a href="https://space.bilibili.com/1595729670">点击访问</a>')
        lbl_bilibili.setTextFormat(Qt.RichText)
        lbl_bilibili.setOpenExternalLinks(True)
        
        # Douyin
        lbl_douyin = QLabel('<b>抖音主页：</b><a href="https://www.douyin.com/user/self?from_tab_name=main">点击访问</a>')
        lbl_douyin.setTextFormat(Qt.RichText)
        lbl_douyin.setOpenExternalLinks(True)

        # GitHub
        lbl_github = QLabel('<b>GitHub主页：</b><a href="https://github.com/luohuabuxiema">点击访问</a>')
        lbl_github.setTextFormat(Qt.RichText)
        lbl_github.setOpenExternalLinks(True)
        
        layout.addWidget(lbl_author)
        layout.addWidget(lbl_csdn)
        layout.addWidget(lbl_bilibili)
        layout.addWidget(lbl_douyin)
        layout.addWidget(lbl_github)
        layout.addStretch()

        btn_layout = QHBoxLayout()
        btn_close = QPushButton("关闭")
        btn_close.setFixedWidth(100)
        btn_close.clicked.connect(self.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)