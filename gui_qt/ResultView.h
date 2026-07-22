#pragma once

#include <QWidget>
#include <QString>

class QLabel;
class QTreeWidget;
class QTreeWidgetItem;
class QPushButton;

namespace fc::gui {

// 结果 JSON 查看器：左右分栏，左侧关键字段高亮，右侧完整树
class ResultView : public QWidget {
    Q_OBJECT
public:
    explicit ResultView(QWidget* parent = nullptr);

public slots:
    void loadFile(const QString& path);
    void clear();

signals:
    void statusMessage(QString msg);

private:
    void buildKeyPanel();
    void buildTreePanel();
    void populateKeyFieldsForCamera(const QJsonObject& obj);
    void populateKeyFieldsForLaser(const QJsonObject& obj);

    // —— 左侧关键字段（每行：标题 + 值）
    QLabel* fileLabel_       = nullptr;
    QLabel* schemaLabel_     = nullptr;
    QLabel* k1Label_         = nullptr;  // 通用 K1（每模块自定）
    QLabel* k2Label_         = nullptr;
    QLabel* k3Label_         = nullptr;
    QLabel* k4Label_         = nullptr;
    QLabel* k5Label_         = nullptr;

    QTreeWidget* tree_       = nullptr;
    QPushButton* openBtn_    = nullptr;
};

}  // namespace fc::gui
