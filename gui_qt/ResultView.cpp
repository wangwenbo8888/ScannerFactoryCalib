#include "ResultView.h"

#include <QFile>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>
#include <QJsonValue>
#include <QLabel>
#include <QPushButton>
#include <QFileDialog>
#include <QTreeWidget>
#include <QVBoxLayout>

namespace fc::gui {

namespace {
QString jsonValToString(const QJsonValue& v) {
    if (v.isString())  return v.toString();
    if (v.isBool())    return v.toBool() ? QStringLiteral("true") : QStringLiteral("false");
    if (v.isDouble())  return QString::number(v.toDouble());
    if (v.isNull())    return QStringLiteral("(null)");
    return v.toString();
}

void fillItem(QTreeWidgetItem* item, const QJsonValue& v) {
    if (v.isObject()) {
        auto obj = v.toObject();
        for (auto it = obj.begin(); it != obj.end(); ++it) {
            auto* child = new QTreeWidgetItem(QStringList{ it.key(), "" });
            fillItem(child, it.value());
            item->addChild(child);
        }
    } else if (v.isArray()) {
        auto arr = v.toArray();
        item->setText(1, QStringLiteral("[%1 items]").arg(arr.size()));
        for (int i = 0; i < arr.size(); ++i) {
            auto* child = new QTreeWidgetItem(QStringList{ QStringLiteral("[%1]").arg(i), "" });
            fillItem(child, arr.at(i));
            item->addChild(child);
        }
    } else {
        item->setText(1, jsonValToString(v));
    }
}
}  // namespace

ResultView::ResultView(QWidget* parent) : QWidget(parent) {
    auto* root = new QHBoxLayout(this);
    root->setContentsMargins(4, 4, 4, 4);

    buildTreePanel();

    auto* left = new QWidget;
    auto* llay = new QVBoxLayout(left);
    llay->setContentsMargins(0, 0, 0, 0);
    openBtn_ = new QPushButton(QStringLiteral("打开结果 JSON..."));
    connect(openBtn_, &QPushButton::clicked, this, [this]() {
        QString p = QFileDialog::getOpenFileName(
            this, QStringLiteral("选择结果 JSON"),
            QString(), QStringLiteral("JSON (*.json)"));
        if (!p.isEmpty()) loadFile(p);
    });
    fileLabel_   = new QLabel(QStringLiteral("<未加载>"));
    schemaLabel_ = new QLabel(QStringLiteral("-"));
    k1Label_     = new QLabel(QStringLiteral("-"));
    k2Label_     = new QLabel(QStringLiteral("-"));
    k3Label_     = new QLabel(QStringLiteral("-"));
    k4Label_     = new QLabel(QStringLiteral("-"));
    k5Label_     = new QLabel(QStringLiteral("-"));

    fileLabel_->setWordWrap(true);
    auto addRow = [&](const QString& title, QLabel* val) {
        auto* row = new QHBoxLayout;
        auto* t = new QLabel(title);
        t->setMinimumWidth(110);
        t->setStyleSheet("font-weight:bold;");
        row->addWidget(t);
        row->addWidget(val, 1);
        auto* w = new QWidget;
        w->setLayout(row);
        llay->addWidget(w);
    };
    llay->addWidget(openBtn_);
    addRow(QStringLiteral("文件"), fileLabel_);
    addRow(QStringLiteral("schema"), schemaLabel_);
    addRow(QStringLiteral("关键字段 1"), k1Label_);
    addRow(QStringLiteral("关键字段 2"), k2Label_);
    addRow(QStringLiteral("关键字段 3"), k3Label_);
    addRow(QStringLiteral("关键字段 4"), k4Label_);
    addRow(QStringLiteral("关键字段 5"), k5Label_);
    llay->addStretch(1);

    root->addWidget(left, 1);
    root->addWidget(tree_, 2);
}

void ResultView::buildTreePanel() {
    tree_ = new QTreeWidget;
    tree_->setHeaderLabels({ QStringLiteral("字段"), QStringLiteral("值") });
    tree_->header()->setStretchLastSection(true);
}

void ResultView::clear() {
    tree_->clear();
    fileLabel_->setText(QStringLiteral("<未加载>"));
    schemaLabel_->setText(QStringLiteral("-"));
    for (auto* l : { k1Label_, k2Label_, k3Label_, k4Label_, k5Label_ }) {
        l->setText(QStringLiteral("-"));
    }
}

void ResultView::loadFile(const QString& path) {
    clear();
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) {
        fileLabel_->setText(QStringLiteral("打开失败: %1").arg(path));
        emit statusMessage(QStringLiteral("打开失败"));
        return;
    }
    QJsonParseError err{};
    auto doc = QJsonDocument::fromJson(f.readAll(), &err);
    if (err.error != QJsonParseError::NoError) {
        fileLabel_->setText(QStringLiteral("解析失败: %1").arg(err.errorString()));
        emit statusMessage(QStringLiteral("JSON 解析失败"));
        return;
    }
    fileLabel_->setText(QFileInfo(path).fileName());

    QJsonObject rootObj = doc.isObject() ? doc.object() : QJsonObject{};
    auto* rootItem = new QTreeWidgetItem(QStringList{ QStringLiteral("<root>"), "" });
    QJsonValue rootVal = doc.isObject()
        ? QJsonValue(doc.object())
        : QJsonValue(doc.array());
    fillItem(rootItem, rootVal);
    tree_->addTopLevelItem(rootItem);
    tree_->expandItem(rootItem);
    tree_->resizeColumnToContents(0);

    // schema
    schemaLabel_->setText(rootObj.value("schema").toString());

    // 按模块自动选关键字段
    QString schema = schemaLabel_->text();
    if (schema.contains(QStringLiteral("laser")) ||
        rootObj.contains("posesProcessed")) {
        populateKeyFieldsForLaser(rootObj);
    } else {
        populateKeyFieldsForCamera(rootObj);
    }
    emit statusMessage(QStringLiteral("已加载: %1").arg(QFileInfo(path).fileName()));
}

void ResultView::populateKeyFieldsForCamera(const QJsonObject& obj) {
    // imageSize
    QJsonValue img = obj.value("imageSize");
    if (img.isArray()) {
        auto a = img.toArray();
        k1Label_->setText(QStringLiteral("imageSize: %1 × %2")
                              .arg(a.at(0).toInt()).arg(a.at(1).toInt()));
    }
    // intrinsic.left.rms
    auto intr = obj.value("intrinsic").toObject();
    auto left = intr.value("left").toObject();
    k2Label_->setText(QStringLiteral("left RMS: %1").arg(left.value("rms").toDouble()));
    auto right = intr.value("right").toObject();
    k3Label_->setText(QStringLiteral("right RMS: %1").arg(right.value("rms").toDouble()));
    // referenceTemp + cte
    k4Label_->setText(QStringLiteral("referenceTemp: %1").arg(obj.value("referenceTemp").toDouble()));
    QJsonValue cte = obj.value("cte");
    if (cte.isDouble()) k5Label_->setText(QStringLiteral("CTE: %1").arg(cte.toDouble()));
}

void ResultView::populateKeyFieldsForLaser(const QJsonObject& obj) {
    k1Label_->setText(QStringLiteral("posesProcessed: %1").arg(obj.value("posesProcessed").toInt()));
    k2Label_->setText(QStringLiteral("framesOk: %1").arg(obj.value("framesOk").toInt()));
    k3Label_->setText(QStringLiteral("accumulatedPoints3D: %1").arg(obj.value("accumulatedPoints3D").toInt()));
    auto build = obj.value("build").toObject();
    k4Label_->setText(QStringLiteral("build: %1").arg(build.value("success").toBool() ? "OK" : "PARTIAL"));
    k5Label_->setText(QStringLiteral("schema: %1").arg(obj.value("schema").toString()));
}

}  // namespace fc::gui
